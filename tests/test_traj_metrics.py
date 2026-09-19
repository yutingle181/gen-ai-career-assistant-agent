"""轨迹级评测（P1-5）：成功口径、Failure Onset、代价指标与报告章节。

为什么值得单独测：这些口径一旦写错，报告会给出「看起来精确」的错误结论
（例如把工具降级算成成功），而这类错误靠肉眼看数字是发现不了的。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import config
from src.eval.trajectory import (
    TrajectorySample,
    failure_onset,
    reached_step_limit,
    render_trajectory,
    stats_from_outcomes,
    success_of,
    summarize,
)


def _ev(name: str = "search_web", ok: bool = True, kind: str = "") -> dict:
    return {"name": name, "ok": ok, "error_kind": kind, "elapsed_ms": 10}


# ------------------------------------------------------------------ 成功口径
def test_success_of_rejects_empty_and_failure_fallbacks():
    assert success_of("qa", "")[0] is False
    assert success_of("qa", "   ")[0] is False

    ok, reason = success_of("qa", "（模型调用失败，请检查 .env 配置后重试。错误详情：TypeError）")
    assert ok is False and "兜底" in reason


def test_success_of_requires_mode_markers():
    """场景结构标记缺失即判失败：各场景的输出契约不同，不能用同一把尺子。"""
    assert success_of("knowledge", "这是一段没有任何引用的回答")[0] is False
    assert success_of("knowledge", "结论来自资料 [1]")[0] is True

    assert success_of("jd_match", "总分 78 分")[0] is False
    assert success_of("jd_match", "## JD 匹配诊断：78/100\n详情…")[0] is True


def test_success_of_treats_degraded_retrieval_as_failure_for_dependent_modes():
    """依赖检索的场景：工具降级时文本再工整也不算成功。"""
    ok, reason = success_of(
        "knowledge", "结论 [1]", [_ev("knowledge_search", ok=False, kind="timeout")]
    )
    assert ok is False and "工具降级" in reason

    # 不依赖检索的场景：一次工具失败不直接判失败（模型可以基于已有信息作答）
    assert success_of("qa", "基于已有知识的回答", [_ev(ok=False, kind="timeout")])[0] is True


# ------------------------------------------------------------------ Failure Onset
def test_failure_onset_returns_first_failing_step():
    events = [_ev(), _ev(ok=False, kind="timeout"), _ev(ok=False, kind="timeout")]
    assert failure_onset(events) == 2
    assert failure_onset([_ev(), _ev()]) is None
    assert failure_onset([]) is None


def test_reached_step_limit_detects_cap_and_recursion_fallback(monkeypatch):
    monkeypatch.setattr(config, "TOOL_CALLING_MAX_STEPS", 2)
    assert reached_step_limit(TrajectorySample(events=[_ev(), _ev()])) is True
    assert reached_step_limit(TrajectorySample(events=[_ev()])) is False

    fallback = TrajectorySample(text="（本轮工具调用已达执行上限，我先基于现有信息给出结论……）")
    assert reached_step_limit(fallback) is True


# ------------------------------------------------------------------ 汇总
def test_summarize_aggregates_rates_and_history():
    samples = [
        TrajectorySample(mode="knowledge", text="答案 [1]", events=[_ev(), _ev()], rounds=3),
        TrajectorySample(
            mode="knowledge", text="答案 [1]", events=[_ev(ok=False, kind="timeout")], rounds=2
        ),
        TrajectorySample(mode="qa", text="", events=[], rounds=1),
    ]
    stats = summarize(samples, max_steps=2)

    assert stats.samples == 3 and stats.success == 1
    assert stats.success_rate == pytest.approx(1 / 3, abs=1e-4)
    assert stats.tool_failure_rate == pytest.approx(1 / 3, abs=1e-4)
    assert stats.over_step_rate == pytest.approx(1 / 3, abs=1e-4)  # 第 1 条撞到上限
    assert stats.avg_rounds == 2.0
    assert stats.failure_onset_hist == {1: 1}
    assert stats.avg_failure_onset == 1.0
    assert len(stats.failed_reasons) == 2  # 工具降级 + 空回答
    assert stats.onset_examples and "第 1 步" in stats.onset_examples[0]


def test_summarize_empty_input_returns_zeros():
    stats = summarize([])
    assert stats.samples == 0 and stats.success_rate == 0.0 and stats.avg_failure_onset is None


def test_stats_from_outcomes_uses_duck_typing():
    """适配器只靠属性取值：评测模块不该反向依赖执行器类型。"""
    outcomes = [
        SimpleNamespace(text="答案 [1]", events=[_ev()], rounds=2, latency_ms=100, total_tokens=50),
        SimpleNamespace(text="", events=[], rounds=1, latency_ms=10, total_tokens=5),
    ]
    stats = stats_from_outcomes(outcomes, mode="knowledge", max_steps=3)
    assert stats.samples == 2 and stats.success == 1


# ------------------------------------------------------------------ 报告章节
def test_render_trajectory_contains_key_sections():
    """直方图/示例这些分支都要渲染（曾因排序 lambda 写错而静默漏掉）。"""
    stats = summarize(
        [
            TrajectorySample(mode="qa", text="", events=[]),
            TrajectorySample(
                mode="knowledge", text="答案 [1]", events=[_ev(ok=False, kind="timeout")], rounds=1
            ),
        ],
        max_steps=2,
    )
    text = "\n".join(render_trajectory(stats, 4))

    assert "## 4. 轨迹级评测" in text
    assert "TaskSuccessRate" in text and "Failure Onset" in text
    assert "失败原因分布" in text
    assert "Failure Onset 直方图（第几步）" in text and "第 1 步：1 条" in text
    assert "最早失败示例" in text

    assert "无有效轨迹样本" in "\n".join(render_trajectory({}, 4))


def test_report_includes_trajectory_from_tool_comparison():
    """A/B 结果里带轨迹指标时，报告必须自动多出一章（调用方无需改代码）。"""
    from src.eval.report import render_report
    from src.eval.tool_eval import ToolComparison, ToolPathMetric
    from src.models import MetricResult

    comparison = ToolComparison(
        explicit=ToolPathMetric(name="显式检索"),
        tool_calling=ToolPathMetric(name="Function Calling"),
        sample_count=1,
        trajectory=summarize(
            [TrajectorySample(mode="qa", text="回答内容", rounds=2)], max_steps=2
        ).to_dict(),
    )
    report = render_report([MetricResult(name="实验")], tool_comparison=comparison)

    assert "轨迹级评测" in report and "TaskSuccessRate" in report
    assert '"trajectory"' in report  # 配置快照里也留档，便于复现
