"""装配 LangGraph 工作流。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..logging_setup import get_logger
from ..state import State
from .nodes import (
    ask_query_bot,
    categorize,
    fallback,
    handle_interview_preparation,
    handle_learning_resource,
    handle_resume_making,
    interview_review,
    interview_topics_questions,
    jd_match,
    job_search,
    knowledge_qa,
    mock_interview,
    route_interview,
    route_learning,
    route_query,
    tutorial_agent,
)

logger = get_logger(__name__)


def build_app():
    """构建并编译工作流。"""
    workflow = StateGraph(State)

    # 分类节点
    workflow.add_node("categorize", categorize)
    workflow.add_node("handle_learning_resource", handle_learning_resource)
    workflow.add_node("handle_interview_preparation", handle_interview_preparation)

    # 业务叶子节点
    workflow.add_node("handle_resume_making", handle_resume_making)
    workflow.add_node("job_search", job_search)
    workflow.add_node("knowledge_qa", knowledge_qa)
    workflow.add_node("tutorial_agent", tutorial_agent)
    workflow.add_node("ask_query_bot", ask_query_bot)
    workflow.add_node("interview_topics_questions", interview_topics_questions)
    workflow.add_node("mock_interview", mock_interview)
    workflow.add_node("jd_match", jd_match)
    workflow.add_node("interview_review", interview_review)
    workflow.add_node("fallback", fallback)

    workflow.add_edge(START, "categorize")

    workflow.add_conditional_edges(
        "categorize",
        route_query,
        {
            "handle_learning_resource": "handle_learning_resource",
            "handle_resume_making": "handle_resume_making",
            "handle_interview_preparation": "handle_interview_preparation",
            "job_search": "job_search",
            "knowledge_qa": "knowledge_qa",
            "jd_match": "jd_match",
            "interview_review": "interview_review",
            "fallback": "fallback",
        },
    )

    workflow.add_conditional_edges(
        "handle_learning_resource",
        route_learning,
        {"tutorial_agent": "tutorial_agent", "ask_query_bot": "ask_query_bot"},
    )

    workflow.add_conditional_edges(
        "handle_interview_preparation",
        route_interview,
        {
            "mock_interview": "mock_interview",
            "interview_topics_questions": "interview_topics_questions",
        },
    )

    for node in (
        "handle_resume_making",
        "job_search",
        "knowledge_qa",
        "tutorial_agent",
        "ask_query_bot",
        "interview_topics_questions",
        "mock_interview",
        "jd_match",
        "interview_review",
        "fallback",
    ):
        workflow.add_edge(node, END)

    workflow.set_entry_point("categorize")
    return workflow.compile()


_app = None


def get_app():
    """获取（并缓存）编译后的工作流。"""
    global _app
    if _app is None:
        _app = build_app()
        logger.info("LangGraph 工作流编译完成")
    return _app


def route_only(query: str) -> dict:
    """只跑分类路由，返回 {"mode": ..., "message": ..., "category": ...}。

    分类失败时不影响主流程：返回 qa 模式作为安全默认。
    """
    try:
        return get_app().invoke({"query": query})
    except Exception as exc:  # noqa: BLE001
        logger.warning("路由失败，回退为问答模式：%s", exc)
        return {"mode": "qa", "message": "（路由异常，已回退为问答模式）", "category": "learning"}


def graph_mermaid() -> str:
    """导出 Mermaid 图文本，供 README / 界面展示。"""
    try:
        return get_app().get_graph().draw_mermaid()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Mermaid 导出失败：%s", exc)
        return ""
