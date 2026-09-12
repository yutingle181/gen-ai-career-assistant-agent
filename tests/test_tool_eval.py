"""双路径 A/B 评测测试：汇总口径、失败容错、真实 runner 与报告章节（全离线）。

设计要点：`compare_paths` 接受**注入的 runner**，因此这里用伪 runner 就能把
「延迟 / token / 轮次 / 成功率」这套口径跑全，不需要任何 API Key 或网络。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.eval.report import render_report, render_tool_comparison
from src.eval.tool_eval import (
    PATH_EXPLICIT,
    PATH_TOOL_CALLING,
    PathOutcome,
    compare_paths,
    count_tokens,
    make_explicit_runner,
    make_tool_runner,
    summarize_path,
)
from src.models import EvalRecord, MetricResult


# ---------------------------------------------------------------- 口径
def test_count_tokens_is_reused_from_cache():
    """成本口径必须与 cache.py 一致，避免两套数字打架。"""
    from src.cache import count_tokens as cache_count

    assert count_tokens is cache_count
    assert count_tokens("") == 0
    assert count_tokens("LangGraph 工具调用") > 0


def test_summarize_path_empty_input():
    metric = summarize_path("空路径", [])
    assert metric.samples == 0
    assert metric.success_rate == 0.0
    assert metric.avg_latency_ms == 0.0
    assert metric.p95_latency_ms == 0.0


def test_summarize_path_aggregates_and_counts_failures():
    outcomes = [
        PathOutcome(latency_ms=100, prompt_tokens=10, completion_tokens=5, rounds=2, tool_calls=1),
        PathOutcome(latency_ms=200, prompt_tokens=20, completion_tokens=10, rounds=3, tool_calls=2),
        PathOutcome(latency_ms=300, prompt_tokens=30, completion_tokens=15, rounds=1, tool_calls=0),
        PathOutcome(ok=False, latency_ms=999, prompt_tokens=999),
    ]
    m = summarize_path(PATH_TOOL_CALLING, outcomes)

    assert m.samples == 4 and m.failures == 1
    assert m.success_rate == 0.75
    # 失败样本不参与延迟 / token / 轮次统计
    assert m.avg_latency_ms == 200.0
    assert m.total_tokens == (15 + 30 + 45)
    assert m.avg_tokens == 30.0
    assert m.avg_rounds == 2.0
    assert m.avg_tool_calls == 1.0
    assert m.tool_call_rate == round(2 / 3, 4)


def test_p95_reports_tail_not_mean():
    outcomes = [PathOutcome(latency_ms=v) for v in (10, 10, 10, 10, 10, 500)]
    assert summarize_path("长尾", outcomes).p95_latency_ms == 500.0


# ---------------------------------------------------------------- 对比
def _record(i: int) -> EvalRecord:
    return EvalRecord(question=f"q{i}", golden_chunk_ids=[f"c{i}"])


def test_compare_paths_with_injected_runners():
    def explicit(rec):
        return PathOutcome(latency_ms=100, prompt_tokens=50, completion_tokens=10, rounds=1, tool_calls=1)

    def tool(rec):
        return PathOutcome(latency_ms=250, prompt_tokens=40, completion_tokens=30, rounds=3, tool_calls=2)

    cmp = compare_paths([_record(i) for i in range(3)], explicit, tool, max_workers=1)

    assert cmp.sample_count == 3
    assert cmp.explicit.name == PATH_EXPLICIT
    assert cmp.tool_calling.name == PATH_TOOL_CALLING
    assert cmp.explicit.avg_latency_ms == 100.0
    assert cmp.tool_calling.avg_tool_calls == 2.0
    assert cmp.delta("avg_latency_ms") == 150.0
    # 单样本 60 -> 70 token，3 条共多 30
    assert cmp.delta("total_tokens") == 30
    assert [label for label, _ in cmp.rows()] == [PATH_EXPLICIT, PATH_TOOL_CALLING]


def test_compare_paths_counts_runner_exceptions_as_failures():
    def boom(rec):
        raise RuntimeError("网络挂了")

    def ok(rec):
        return PathOutcome(latency_ms=10, rounds=1)

    def wrong_type(rec):
        return "不是 PathOutcome"

    cmp = compare_paths([_record(0), _record(1)], boom, wrong_type, max_workers=1)

    assert cmp.explicit.success_rate == 0.0
    assert cmp.explicit.failures == 2
    assert cmp.tool_calling.success_rate == 0.0  # 类型异常也计为失败，不抛给上层
    assert cmp.tool_calling.samples == 2

    cmp2 = compare_paths([_record(0)], ok, ok, max_workers=1)
    assert cmp2.explicit.success_rate == 1.0


# ---------------------------------------------------------------- 真实 runner
class _Chunk:
    def __init__(self, cid: str, text: str) -> None:
        self.chunk_id = cid
        self.text = text


class _FakePipeline:
    def retrieve(self, query, cfg=None, use_cache=True):
        assert use_cache is False, "评测必须跳过召回缓存，否则延迟被预热打平"
        return [_Chunk("c1", "片段一内容"), _Chunk("c2", "片段二内容")]


def test_explicit_runner_uses_single_retrieval_and_counts_tokens():
    captured: dict = {}

    def fake_llm(messages, model=None):
        captured["messages"] = messages
        captured["model"] = model
        return "固定答案"

    runner = make_explicit_runner(_FakePipeline(), llm=fake_llm, model="qwen-turbo")
    out = runner(_record(0))

    assert out.ok is True
    assert out.rounds == 1
    assert out.tool_calls == 1
    assert out.citations == ["c1", "c2"]
    assert out.prompt_tokens > 0 and out.completion_tokens > 0
    assert out.latency_ms >= 0
    assert "【检索结果】" in captured["messages"][1].content
    assert captured["model"] == "qwen-turbo"  # 两条路径必须钉在同一模型上才公平


def test_tool_runner_derives_rounds_and_calls_from_events():
    emitted: list[dict] = []

    def fake_run_with_tools(messages, tools=None, model=None, max_tokens=None, max_steps=None, on_event=None):
        for name in ("web_search", "kb_search"):
            payload = {"name": name, "ok": True, "elapsed_ms": 12}
            emitted.append(payload)
            if on_event:
                on_event(payload)
        return "工具调用后的最终答案"

    runner = make_tool_runner([object()], runner=fake_run_with_tools)
    out = runner(_record(0))

    assert out.ok is True
    # 首轮决策 + 2 次工具回灌后各一轮 = 3 轮
    assert out.rounds == 3
    assert out.tool_calls == 2
    assert len(emitted) == 2
    assert out.completion_tokens > 0


# ---------------------------------------------------------------- 报告
def _comparison():
    def explicit(rec):
        return PathOutcome(latency_ms=800, prompt_tokens=300, completion_tokens=100, rounds=1, tool_calls=1)

    def tool(rec):
        return PathOutcome(latency_ms=2600, prompt_tokens=200, completion_tokens=150, rounds=3, tool_calls=2)

    return compare_paths([_record(i) for i in range(2)], explicit, tool, max_workers=1)


def test_render_tool_comparison_section():
    lines = render_tool_comparison(_comparison())
    md = "\n".join(lines)

    assert "工具调用 A/B 对比" in md
    assert f"TOOL_CALLING_MAX_STEPS={config.TOOL_CALLING_MAX_STEPS}" in md
    assert PATH_EXPLICIT in md and PATH_TOOL_CALLING in md
    assert "结论（全部来自上表实测）" in md
    assert "取舍建议" in md


def test_render_report_appends_tool_section_and_renumbers(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", Path(tmp_path))
    results = [
        MetricResult(name="混合检索(RRF)", recall_at_k=0.8, mrr=0.7, hit_rate=0.9, sample_count=5)
    ]

    plain = render_report(results, None, k=5)
    assert "## 3. 结论与优化建议" in plain
    assert "工具调用 A/B 对比" not in plain

    with_tool = render_report(results, None, k=5, tool_comparison=_comparison())
    assert "## 3. 工具调用 A/B 对比" in with_tool
    assert "## 4. 结论与优化建议" in with_tool
    assert "## 5. 配置快照" in with_tool
    assert "tool_comparison" in with_tool


def test_tool_conclusions_handle_empty_samples():
    from src.eval.report import _tool_conclusions
    from src.eval.tool_eval import ToolComparison, ToolPathMetric

    empty = ToolComparison(
        explicit=ToolPathMetric(name=PATH_EXPLICIT),
        tool_calling=ToolPathMetric(name=PATH_TOOL_CALLING),
    )
    assert _tool_conclusions(empty) == ["- 无有效样本，无法对比两条路径。"]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """兜底：任何用例都不应真的联网。"""
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", False)
    yield
