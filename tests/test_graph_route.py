"""路由与图结构测试（不依赖真实 API）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph.nodes import (  # noqa: E402
    _heuristic_category,
    route_interview,
    route_learning,
    route_query,
)
from src.graph.workflow import build_app, graph_mermaid, route_only  # noqa: E402
from src.state import (  # noqa: E402
    CAT_INTERVIEW,
    CAT_INTERVIEW_REVIEW,
    CAT_JD_MATCH,
    CAT_JOB_SEARCH,
    CAT_KNOWLEDGE,
    CAT_LEARNING,
    CAT_RESUME,
    MODE_KNOWLEDGE,
)


def test_heuristic_category():
    cases = [
        ("帮我写一篇 LangGraph 教程", CAT_LEARNING),
        ("帮我改一下简历", CAT_RESUME),
        ("AI 工程师面试会问什么", CAT_INTERVIEW),
        ("长沙有没有 AI 岗位在招", CAT_JOB_SEARCH),
        ("根据我上传的文档回答", CAT_KNOWLEDGE),
        # 新增场景：更具体的意图必须优先于宽泛的面试/简历规则
        ("帮我复盘刚才的面试", CAT_INTERVIEW_REVIEW),
        ("我的简历和这个 JD 匹配度怎么样", CAT_JD_MATCH),
    ]
    for query, expected in cases:
        got, _ = _heuristic_category(query)
        assert got == expected, f"{query} 期望 {expected}，实际 {got}"
    print("[OK] 关键词兜底分类正确")


def test_heuristic_specific_intent_beats_broad_rule():
    """「复盘」「匹配度」不应被宽泛的面试/简历规则截胡。"""
    assert _heuristic_category("这场面试我哪里答得不好，复盘一下")[0] == CAT_INTERVIEW_REVIEW
    assert _heuristic_category("这份简历跟岗位要求哪里不匹配")[0] == CAT_JD_MATCH
    # 裸 "jd" 不触发匹配场景，避免「根据 JD 改写简历」被误判
    assert _heuristic_category("根据这个 JD 改一下我的简历")[0] == CAT_RESUME


def test_route_functions():
    assert route_query({"category": "learning"}) == "handle_learning_resource"
    assert route_query({"category": "resume"}) == "handle_resume_making"
    assert route_query({"category": "interview"}) == "handle_interview_preparation"
    assert route_query({"category": "job_search"}) == "job_search"
    assert route_query({"category": "knowledge"}) == "knowledge_qa"
    assert route_query({"category": "jd_match"}) == "jd_match"
    assert route_query({"category": "interview_review"}) == "interview_review"
    assert route_query({"category": "unknown"}) == "fallback"

    assert route_learning({"category": "tutorial"}) == "tutorial_agent"
    assert route_learning({"category": "question"}) == "ask_query_bot"
    assert route_learning({"category": "乱码"}) == "ask_query_bot"

    assert route_interview({"category": "mock"}) == "mock_interview"
    assert route_interview({"category": "question"}) == "interview_topics_questions"
    assert route_interview({"category": "乱码"}) == "interview_topics_questions"
    print("[OK] 一二级路由函数全部有出口（含兜底）")


def test_graph_compile():
    app = build_app()
    assert app is not None
    mermaid = graph_mermaid()
    assert "categorize" in mermaid
    print("[OK] 图编译成功，Mermaid 可导出")


def test_route_only_offline():
    """无 API Key 时应走兜底而非抛异常。"""
    result = route_only("帮我改一下简历")
    assert result.get("mode"), "route_only 必须返回 mode"
    print(f"[OK] route_only 可用 -> mode={result.get('mode')} category={result.get('category')}")

    result2 = route_only("根据我上传的文档，RAG 切分怎么做的？")
    print(f"[OK] 知识库路由 -> mode={result2.get('mode')} (期望 {MODE_KNOWLEDGE})")


if __name__ == "__main__":
    test_heuristic_category()
    test_route_functions()
    test_graph_compile()
    test_route_only_offline()
    print("\n全部路由测试通过")
