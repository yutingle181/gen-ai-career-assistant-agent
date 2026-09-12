"""Streamlit 新增可视模块的渲染测试（纯字符串，无需启动 Streamlit）。

重点验证两件事：
1. 该有的信息都在（工具名 / 耗时 / 分项 / 命中缺口 / 面试重点）；
2. 用户与工具产出的文本都经过转义——工具返回内容来自外部，直接拼 HTML 会产生注入风险。
"""

from __future__ import annotations

from src.models import JDMatchResult
from src.ui_style import jd_scorecard_html, tool_timeline_html

DIMS = {"skills": "技能匹配", "experience": "经验匹配"}


# ---------------------------------------------------------------- 时间线
def test_timeline_empty_state():
    html = tool_timeline_html([])
    assert "未调用工具，直接作答" in html
    assert "tl-step" not in html


def test_timeline_renders_each_step():
    events = [
        {
            "name": "search_web",
            "ok": True,
            "elapsed_ms": 812,
            "args_summary": '{"query": "长沙 AI 岗位"}',
            "result_summary": "3 条结果",
        },
        {
            "name": "knowledge_search",
            "ok": False,
            "elapsed_ms": 120,
            "args_summary": '{"query": "切分策略"}',
            "result_summary": "工具执行失败",
        },
    ]
    html = tool_timeline_html(events)

    assert "共 2 步 · 合计 932ms" in html
    assert "search_web" in html and "knowledge_search" in html
    assert "#1 · 812ms" in html and "#2 · 120ms" in html
    assert 'tl-dot ok' in html and 'tl-dot fail' in html
    assert html.count("tl-step") == 2


def test_timeline_escapes_tool_output():
    """工具返回来自外部，必须转义，否则可注入脚本。"""
    html = tool_timeline_html(
        [
            {
                "name": "search_web",
                "ok": True,
                "elapsed_ms": 10,
                "args_summary": "<img src=x onerror=alert(1)>",
                "result_summary": "<script>alert(1)</script>",
            }
        ]
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "onerror=alert(1)" in html  # 作为纯文本保留，但标签被转义
    assert "<img src=x" not in html


# ---------------------------------------------------------------- 评分卡
def _result(**kwargs) -> JDMatchResult:
    base = {
        "total_score": 78,
        "dimension_scores": {"skills": 80, "experience": 70},
        "matched": ["Python", "LangChain"],
        "gaps": ["Kubernetes"],
        "interview_focus": ["准备分布式部署案例"],
        "summary": "整体匹配良好。",
    }
    base.update(kwargs)
    return JDMatchResult(**base)


def test_scorecard_renders_ring_dims_and_columns():
    html = jd_scorecard_html(_result(), DIMS)

    assert "conic-gradient(#22D3EE 0% 78%, rgba(255,255,255,.09) 78% 100%)" in html
    assert '<span class="score-num">78</span>' in html
    assert "技能匹配" in html and "经验匹配" in html
    assert 'style="width:80%"' in html and 'style="width:70%"' in html
    assert "✅ 命中项" in html and "Python" in html and "LangChain" in html
    assert "🟠 缺口项" in html and "Kubernetes" in html
    assert "面试准备重点" in html and "准备分布式部署案例" in html
    assert "整体匹配良好。" in html


def test_scorecard_falls_back_for_unknown_dimension_key():
    html = jd_scorecard_html(_result(dimension_scores={"custom_dim": 55}), DIMS)
    assert "custom_dim" in html  # 未知维度直接显示原始 key，不丢分项


def test_scorecard_empty_lists_have_explicit_copy():
    html = jd_scorecard_html(_result(matched=[], gaps=[], interview_focus=[]), DIMS)
    assert "暂无命中项" in html
    assert "暂无明显缺口" in html
    assert "focus-strip" not in html  # 无重点时不渲染空条


def test_scorecard_clamps_out_of_range_scores():
    """模型侧已有 le=100 校验，这里额外验证渲染层对越界值也不会画出超长进度条。"""

    class _Loose:
        total_score = 180
        dimension_scores = {"skills": 150}
        matched: list[str] = []
        gaps: list[str] = []
        interview_focus: list[str] = []
        summary = ""

    html = jd_scorecard_html(_Loose(), DIMS)
    assert '<span class="score-num">100</span>' in html
    assert 'style="width:100%"' in html


def test_scorecard_escapes_model_output():
    html = jd_scorecard_html(
        _result(summary="<b>加粗</b>", matched=["<script>alert(1)</script>"]), DIMS
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>加粗</b>" not in html
