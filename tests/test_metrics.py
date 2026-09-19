"""运行指标（P2-7）：计数、比率、分布窗口与接口契约。

这些数字的价值在于「当场可见」：排查「为什么这轮没检索」「超轮次是不是变多了」
时不该先去翻日志。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src import metrics


@pytest.fixture(autouse=True)
def _clean():
    metrics.reset()
    yield
    metrics.reset()


@tool
def search_web(query: str) -> str:
    """伪联网检索工具（测试用）。"""
    return f"结果：{query}"


# ------------------------------------------------------------------ 计数与比率
def test_counters_and_rates():
    metrics.incr("tool.call.total", 4)
    metrics.incr("tool.call.fail", 1)
    metrics.incr("agent.run", 2)
    metrics.incr("agent.step_limit", 1)

    snapshot = metrics.snapshot()
    assert snapshot["counters"]["tool.call.total"] == 4
    assert snapshot["rates"]["tool_failure_rate"] == 0.25
    assert snapshot["rates"]["over_step_rate"] == 0.5
    assert snapshot["rates"]["avg_tool_rounds"] == 0.0  # 没记 tool_round 就是 0，不除零


def test_rates_are_zero_without_denominator():
    metrics.incr("tool.call.fail")
    assert metrics.snapshot()["rates"]["tool_failure_rate"] == 0.0


def test_snapshot_only_lists_non_zero_counters():
    metrics.incr("a")
    metrics.incr("b", 0)
    assert metrics.snapshot()["counters"] == {"a": 1}


# ------------------------------------------------------------------ 分布窗口
def test_observe_keeps_recent_window():
    for i in range(250):
        metrics.observe("llm.latency_ms", i)

    dist = metrics.snapshot()["distributions"]["llm.latency_ms"]
    assert dist["n"] == 200  # 只保留最近 200 条，长跑进程不涨内存
    assert dist["p95"] >= dist["avg"]


# ------------------------------------------------------------------ 闭环接入
def test_tool_loop_records_run_rounds_and_ok(monkeypatch):
    """工具闭环执行会写入计数：agent.run 是所有比率的分母。"""
    from src.graph import tool_loop

    class _FakeModel:
        def __init__(self) -> None:
            self._idx = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages, **kwargs):
            self._idx += 1
            if self._idx == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "c1"}],
                )
            return AIMessage(content="最终答案")

    monkeypatch.setattr(tool_loop, "get_chat_model", lambda **kwargs: _FakeModel())
    out = tool_loop.run_with_tools([HumanMessage(content="查")], tools=[search_web])

    assert out == "最终答案"
    assert metrics.counter("agent.run") == 1
    assert metrics.counter("agent.tool_round") == 1
    assert metrics.counter("agent.ok") == 1
    assert metrics.snapshot()["rates"]["avg_tool_rounds"] == 1.0


def test_step_limit_is_counted(monkeypatch):
    from src.graph import tool_loop

    class _LoopingModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages, **kwargs):
            return AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "c1"}],
            )

    monkeypatch.setattr(tool_loop, "get_chat_model", lambda **kwargs: _LoopingModel())
    tool_loop.run_with_tools([HumanMessage(content="查")], tools=[search_web], max_steps=2)

    assert metrics.counter("agent.step_limit") == 1
    assert metrics.snapshot()["rates"]["over_step_rate"] == 1.0


# ------------------------------------------------------------------ 接口契约
def test_health_response_carries_runtime_and_metrics():
    """排查需要的运行时开关与指标，都在 /health 的同一份响应里。"""
    from src.api.schemas import HealthResponse

    model = HealthResponse(
        status="ok",
        llm=False,
        embedding=False,
        runtime={"checkpoint": "memory", "slot_memory": False},
        metrics={"rates": {"tool_failure_rate": 0.1}},
    )
    assert model.runtime["checkpoint"] == "memory"
    assert model.metrics["rates"]["tool_failure_rate"] == 0.1
