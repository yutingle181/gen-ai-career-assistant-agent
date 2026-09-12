"""LangGraph 节点与路由确定性测试（无真实 Key / 网络）。

通过 stub ``get_chat_model`` 与 ``llm_invoke`` 覆盖：
- 一级/二级分类的结构化成功路径、关键词兜底、文本解析；
- 模型/LLM 调用异常时的逐级降级；
- 全部叶子节点、知识库节点（空/非空）、兜底节点；
- 一级与二级路由映射。
"""

from __future__ import annotations

import pytest

from src.graph import nodes
from src.models import CategoryResult, InterviewSubResult, LearningSubResult
from src.state import (
    CAT_FALLBACK,
    CAT_INTERVIEW,
    CAT_INTERVIEW_REVIEW,
    CAT_JD_MATCH,
    CAT_JOB_SEARCH,
    CAT_KNOWLEDGE,
    CAT_LEARNING,
    CAT_RESUME,
    MODE_FALLBACK,
    MODE_INTERVIEW_QUESTIONS,
    MODE_INTERVIEW_REVIEW,
    MODE_JD_MATCH,
    MODE_JOB_SEARCH,
    MODE_KNOWLEDGE,
    MODE_MOCK_INTERVIEW,
    MODE_QA,
    MODE_RESUME,
    MODE_TUTORIAL,
)


class _Struct:
    def __init__(self, holder):
        self._holder = holder

    def invoke(self, messages):
        return self._holder["payload"]


class _Model:
    def __init__(self, holder):
        self._holder = holder

    def with_structured_output(self, schema):
        return _Struct(self._holder)


@pytest.fixture
def fake_llm(monkeypatch):
    import src.llm as llm_mod

    holder = {"payload": None, "text": "tutorial"}

    def _get_chat_model(*a, **k):
        return _Model(holder)

    def _llm_invoke(*a, **k):
        return holder["text"]

    monkeypatch.setattr(llm_mod, "get_chat_model", _get_chat_model)
    monkeypatch.setattr(nodes, "llm_invoke", _llm_invoke)
    return holder


def test_categorize_structured(fake_llm):
    fake_llm["payload"] = CategoryResult(category=CAT_LEARNING, reason="x")
    assert nodes.categorize({"query": "如何学习 RAG"}) == {"category": CAT_LEARNING}


def test_categorize_structured_invalid_type_falls_to_heuristic(fake_llm):
    fake_llm["payload"] = "not a result"
    assert nodes.categorize({"query": "知识库里有什么"}) == {"category": CAT_KNOWLEDGE}


def test_categorize_heuristic_fallback(fake_llm):
    fake_llm["payload"] = None
    assert nodes.categorize({"query": "帮我写一份简历"}) == {"category": CAT_RESUME}


def test_categorize_text_parse_fallback(fake_llm):
    fake_llm["payload"] = None
    fake_llm["text"] = "resume 简历"
    assert nodes.categorize({"query": "随便聊聊"}) == {"category": CAT_RESUME}


def test_structured_falls_back_on_model_error(fake_llm, monkeypatch):
    import src.llm as llm_mod

    fake_llm["payload"] = None
    fake_llm["text"] = "resume 简历"

    class _BoomStruct:
        def invoke(self, messages):
            raise RuntimeError("boom")

    class _BoomModel:
        def with_structured_output(self, schema):
            return _BoomStruct()

    monkeypatch.setattr(llm_mod, "get_chat_model", lambda *a, **k: _BoomModel())
    assert nodes.categorize({"query": "帮我写简历"}) == {"category": CAT_RESUME}


def test_structured_returns_none_on_full_failure(monkeypatch):
    import src.llm as llm_mod

    class _BoomStruct:
        def invoke(self, messages):
            raise RuntimeError("boom")

    class _BoomModel:
        def with_structured_output(self, schema):
            return _BoomStruct()

    llm_mod.get_chat_model = lambda *a, **k: _BoomModel()  # type: ignore[assignment]
    nodes.llm_invoke = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))  # type: ignore[assignment]
    assert nodes.categorize({"query": "帮我写简历"}) == {"category": CAT_RESUME}


def test_categorize_knowledge_without_signal_falls_back(fake_llm):
    """判为知识库却无「已上传资料」指向时，退回启发式，避免进空知识库节点。"""
    fake_llm["payload"] = CategoryResult(category=CAT_KNOWLEDGE, reason="x")
    assert nodes.categorize({"query": "RAG 和微调到底该怎么选？"}) == {"category": CAT_LEARNING}


def test_categorize_knowledge_with_signal_is_kept(fake_llm):
    """明确指向知识库时应保留知识库分类。"""
    fake_llm["payload"] = CategoryResult(category=CAT_KNOWLEDGE, reason="x")
    assert nodes.categorize({"query": "根据知识库回答：切分策略怎么选？"}) == {"category": CAT_KNOWLEDGE}


def test_interview_list_intent_is_deterministic(fake_llm):
    """「整理 N 道面试题」必须走真题清单，不受模型判断影响。"""
    fake_llm["payload"] = InterviewSubResult(sub="mock")
    assert nodes.handle_interview_preparation({"query": "整理 20 道 GenAI 面试题"}) == {"category": "question"}


def test_handle_learning_resource_structured(fake_llm):
    fake_llm["payload"] = LearningSubResult(sub="tutorial")
    assert nodes.handle_learning_resource({"query": "x"}) == {"category": "tutorial"}


def test_handle_learning_resource_heuristic(fake_llm):
    fake_llm["payload"] = None
    assert nodes.handle_learning_resource({"query": "写一份教程"}) == {"category": "tutorial"}


def test_handle_learning_resource_question(fake_llm):
    fake_llm["payload"] = None
    fake_llm["text"] = "中性回答，不含关键词"
    assert nodes.handle_learning_resource({"query": "什么是 RAG"}) == {"category": "question"}


def test_handle_interview_preparation_structured(fake_llm):
    fake_llm["payload"] = InterviewSubResult(sub="mock")
    assert nodes.handle_interview_preparation({"query": "x"}) == {"category": "mock"}


def test_handle_interview_preparation_heuristic(fake_llm):
    fake_llm["payload"] = None
    assert nodes.handle_interview_preparation({"query": "来面我吧"}) == {"category": "mock"}


def test_handle_interview_preparation_question(fake_llm):
    fake_llm["payload"] = None
    fake_llm["text"] = "中性回答，不含关键词"
    assert nodes.handle_interview_preparation({"query": "面试题"}) == {"category": "question"}


def test_leaf_nodes():
    assert nodes.tutorial_agent({})["mode"] == MODE_TUTORIAL
    assert nodes.ask_query_bot({})["mode"] == MODE_QA
    assert nodes.handle_resume_making({})["mode"] == MODE_RESUME
    assert nodes.interview_topics_questions({})["mode"] == MODE_INTERVIEW_QUESTIONS
    assert nodes.mock_interview({})["mode"] == MODE_MOCK_INTERVIEW
    assert nodes.job_search({})["mode"] == MODE_JOB_SEARCH
    assert nodes.jd_match({})["mode"] == MODE_JD_MATCH
    assert nodes.interview_review({})["mode"] == MODE_INTERVIEW_REVIEW


def test_knowledge_qa_no_kb(monkeypatch):
    class _Reg:
        def names(self):
            return []

    monkeypatch.setattr(nodes, "get_registry", lambda: _Reg())
    out = nodes.knowledge_qa({})
    assert out["mode"] == MODE_KNOWLEDGE
    assert "尚未创建" in out["message"]


def test_knowledge_qa_with_kb(monkeypatch):
    class _Reg:
        def names(self):
            return ["kb1", "kb2"]

    monkeypatch.setattr(nodes, "get_registry", lambda: _Reg())
    out = nodes.knowledge_qa({})
    assert "kb1, kb2" in out["message"]


def test_fallback():
    out = nodes.fallback({})
    assert out["mode"] == MODE_FALLBACK
    assert out["category"] == CAT_FALLBACK


def test_route_query():
    assert nodes.route_query({"category": CAT_LEARNING}) == "handle_learning_resource"
    assert nodes.route_query({"category": CAT_RESUME}) == "handle_resume_making"
    assert nodes.route_query({"category": CAT_INTERVIEW}) == "handle_interview_preparation"
    assert nodes.route_query({"category": CAT_JOB_SEARCH}) == "job_search"
    assert nodes.route_query({"category": CAT_KNOWLEDGE}) == "knowledge_qa"
    assert nodes.route_query({"category": CAT_JD_MATCH}) == "jd_match"
    assert nodes.route_query({"category": CAT_INTERVIEW_REVIEW}) == "interview_review"
    assert nodes.route_query({"category": "unknown"}) == "fallback"


def test_route_learning():
    assert nodes.route_learning({"category": "tutorial"}) == "tutorial_agent"
    assert nodes.route_learning({"category": "question"}) == "ask_query_bot"
    assert nodes.route_learning({"category": "other"}) == "ask_query_bot"


def test_route_interview():
    assert nodes.route_interview({"category": "mock"}) == "mock_interview"
    assert nodes.route_interview({"category": "question"}) == "interview_topics_questions"
    assert nodes.route_interview({"category": "other"}) == "interview_topics_questions"


def test_categorize_heuristic_default_learning(fake_llm):
    fake_llm["payload"] = None
    assert nodes.categorize({"query": "今天天气怎么样"}) == {"category": CAT_LEARNING}

