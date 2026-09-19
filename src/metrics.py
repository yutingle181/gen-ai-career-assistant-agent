"""运行指标（P2-7）：进程内计数器与比率，供 /health 暴露与本地排查。

为什么不直接引 Prometheus 客户端：本项目的可观测性主链路是 OpenTelemetry
（可选依赖，见 `telemetry.py`），这里只补一层「没有后端也能看」的轻量计数。
工具失败率、超轮次率这些数字的第一消费者其实是本地排查与评测复核——
它们要的是**当场可见**，而不是先接进大盘。

设计取舍：
- 只提供 `incr` / `observe` / `snapshot` / `reset` 四个入口，够用即止；
- 比率在 `snapshot()` 里算，避免调用方各自写一遍分母；
- 全部加锁：FastAPI 用线程池执行同步代码（`asyncio.to_thread`），
  指标这点开销不值得为了性能做无锁假设。
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any

_LOCK = threading.Lock()
_COUNTERS: Counter[str] = Counter()
_DISTRIBUTIONS: dict[str, list[float]] = {}

#: 需要作为「比率」暴露的指标：(分子, 分母, 指标名)
_RATE_RULES: tuple[tuple[str, str, str], ...] = (
    ("tool.call.fail", "tool.call.total", "tool_failure_rate"),
    ("tool.call.blocked", "tool.call.total", "tool_circuit_blocked_rate"),
    ("agent.step_limit", "agent.run", "over_step_rate"),
    ("agent.recursion_stop", "agent.run", "recursion_stop_rate"),
    ("agent.tool_round", "agent.run", "avg_tool_rounds"),
)

_DISTRIBUTION_KEEP = 200


def incr(name: str, value: int = 1) -> None:
    """计数 +1（或 +value）。"""
    if not name:
        return
    with _LOCK:
        _COUNTERS[name] += value


def observe(name: str, value: float, keep: int = _DISTRIBUTION_KEEP) -> None:
    """记录一次观测值（如耗时），只保留最近 `keep` 条，防止长跑进程吃内存。"""
    if not name:
        return
    with _LOCK:
        series = _DISTRIBUTIONS.setdefault(name, [])
        series.append(float(value))
        if len(series) > keep:
            del series[: len(series) - keep]


def counter(name: str) -> int:
    """读取单个计数（测试与断言用）。"""
    with _LOCK:
        return int(_COUNTERS.get(name, 0))


def rate(numerator: str, denominator: str) -> float:
    """安全比率：分母为 0 时返回 0.0，绝不抛除零。"""
    with _LOCK:
        top = _COUNTERS.get(numerator, 0)
        bottom = _COUNTERS.get(denominator, 0)
    return round(top / bottom, 4) if bottom else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


def snapshot() -> dict[str, Any]:
    """当前指标快照：计数（非零项）+ 比率 + 分布摘要。"""
    with _LOCK:
        counters = {k: int(v) for k, v in _COUNTERS.items() if v}
        series = {k: list(v) for k, v in _DISTRIBUTIONS.items()}

    rates: dict[str, float] = {}
    for top, bottom, label in _RATE_RULES:
        with _LOCK:
            numerator = _COUNTERS.get(top, 0)
            denominator = _COUNTERS.get(bottom, 0)
        rates[label] = round(numerator / denominator, 4) if denominator else 0.0

    distributions = {
        name: {
            "n": len(values),
            "avg": round(sum(values) / len(values), 2) if values else 0.0,
            "p95": _percentile(values, 0.95),
        }
        for name, values in series.items()
    }
    return {"counters": counters, "rates": rates, "distributions": distributions}


def reset() -> None:
    """清空全部指标（测试与压测前重置用）。"""
    with _LOCK:
        _COUNTERS.clear()
        _DISTRIBUTIONS.clear()
