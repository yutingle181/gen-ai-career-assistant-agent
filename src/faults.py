"""故障注入（chaos）：让「失败路径」也能被稳定复现与验证。

动机很直接：失败分级、熔断、Failure Onset 这三块能力在 happy path 上永远是
「0 失败、100% 成功」——那样的报告什么都证明不了。真正要证明的是
「上游坏掉时，系统还能不能给出可读答案、有没有如实把它记下来」。

于是这里提供一个**开关式、可复现**的故障源：
- 默认关闭（`ENABLE_FAULT_INJECTION=false`），关闭时零开销、零行为变化；
- 打开后对 `FAULT_INJECT_TOOL` 指定的工具，以 `FAULT_INJECT_RATE` 的概率
  抛出 `FAULT_INJECT_KIND` 类型的模拟错误（http_5xx / http_4xx / timeout / empty）；
- 固定种子 + 调用序号 ⇒ **同一批输入得到同一串故障**，报告才谈得上可复现；
  注意并发会打乱调用顺序，因此注入实验请用串行（`--workers 1`）。

边界（必须说清）：这是演示与评测用的注入口，**不接真实流量**；
开启时日志会打显眼警告，避免出现「线上莫名失败」却没人知道原因的事故。
"""

from __future__ import annotations

import random
import threading
from typing import Any

from . import config

_LOCK = threading.Lock()
_RNG = random.Random()
_CALLS = 0

#: 模拟故障类型 → 附带的 HTTP 状态码（让 `tools.classify_error` 能走真实分支）
_KIND_STATUS: dict[str, int] = {"http_5xx": 503, "http_4xx": 400}

VALID_KINDS: tuple[str, ...] = ("http_5xx", "http_4xx", "timeout", "empty")


def is_enabled() -> bool:
    """当前是否开启注入（关闭时所有注入口都是无副作用的空操作）。"""
    return bool(config.ENABLE_FAULT_INJECTION and config.FAULT_INJECT_TOOL and config.FAULT_INJECT_RATE > 0)


def targets(tool_name: str) -> bool:
    """本次注入是否命中该工具。"""
    return is_enabled() and tool_name == config.FAULT_INJECT_TOOL


def _roll() -> bool:
    """掷一次骰子（固定种子 → 可复现的故障序列）。"""
    global _CALLS
    with _LOCK:
        if _CALLS == 0:
            _RNG.seed(config.FAULT_INJECT_SEED)
        _CALLS += 1
        return _RNG.random() < config.FAULT_INJECT_RATE


def _simulated_exception(kind: str) -> BaseException:
    """构造与真实故障形态一致的异常（含 status_code，便于被正确分级）。"""
    if kind == "timeout":
        return TimeoutError("fault-injection: simulated upstream timeout")
    exc = RuntimeError(f"fault-injection: simulated upstream error ({kind})")
    status = _KIND_STATUS.get(kind)
    if status is not None:
        exc.status_code = status  # type: ignore[attr-defined]
    return exc


def maybe_fail(tool_name: str) -> BaseException | None:
    """命中注入口时返回一个模拟异常，否则返回 None。

    调用方负责 `raise`：这样异常会走与真实故障**完全同一条**处理链
    （超时守护线程 → 错误分级 → 重试/换源 → 降级标记 → 熔断计数），
    而不是在调用点特判 —— 特判出来的「演示」证明不了任何东西。
    """
    if not targets(tool_name):
        return None
    if config.FAULT_INJECT_KIND == "empty":
        return None  # empty 不是异常，交给 maybe_empty（且不消耗一次掷骰）
    if not _roll():
        return None
    kind = config.FAULT_INJECT_KIND if config.FAULT_INJECT_KIND in VALID_KINDS else "http_5xx"
    return _simulated_exception(kind)


def maybe_empty(tool_name: str) -> bool:
    """是否要模拟「工具正常返回但结果为空」（与「报错」语义完全不同的一种失败）。"""
    if not targets(tool_name) or config.FAULT_INJECT_KIND != "empty":
        return False
    return _roll()


def describe() -> dict[str, Any]:
    """注入配置快照（供日志 / 报告 / 排查共用同一份口径）。"""
    return {
        "enabled": is_enabled(),
        "tool": config.FAULT_INJECT_TOOL,
        "kind": config.FAULT_INJECT_KIND,
        "rate": config.FAULT_INJECT_RATE,
        "seed": config.FAULT_INJECT_SEED,
        "rolls": _CALLS,
    }


def reset() -> None:
    """清空掷骰计数（测试与多次实验之间复用同一串故障序列）。"""
    global _CALLS
    with _LOCK:
        _CALLS = 0
