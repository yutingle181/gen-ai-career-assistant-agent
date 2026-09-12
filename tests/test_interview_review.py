"""面试复盘场景：结构化产出与字段对齐（全离线，无需 API Key）。

覆盖：
- 结构化复核成功 -> 渲染为分区 Markdown（评分 / 追问链 / 薄弱点 / 改进动作）；
- 结构化异常 -> 回退纯文本；
- 缺上下文提示；
- 元数据（不需要人工确认，避免复盘流程被确认弹窗打断）；
- 前端对齐字段（overall / dimensions / question_chain / weaknesses / suggestions / detail）。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.agents import create_agent
from src.agents import interview_review as ir_mod
from src.agents.interview_review import (
    MISSING_CONTEXT_TIP,
    InterviewReviewAgent,
    render_interview_review,
)
from src.agents.jd_match import JDMatchAgent
from src.models import InterviewReview
from src.state import MODE_INTERVIEW_REVIEW


def _sample_review() -> InterviewReview:
    return InterviewReview(
        overall=72,
        dimensions={"技术深度": 70, "表达结构": 78},
        question_chain=["自我介绍", "追问 RAG 切分策略", "项目难点"],
        weaknesses=["对向量检索原理回答含糊"],
        suggestions=["复盘 HNSW 原理", "用 STAR 重写项目描述"],
        detail="整体思路清晰，技术细节需要夯实。",
    )


class _Struct:
    def __init__(self, payload, boom=False):
        self._payload = payload
        self._boom = boom

    def invoke(self, messages):
        if self._boom:
            raise RuntimeError("structured boom")
        return self._payload


class _Model:
    def __init__(self, payload, boom=False):
        self._payload = payload
        self._boom = boom

    def with_structured_output(self, schema):
        return _Struct(self._payload, self._boom)


@pytest.fixture
def patch_llm(monkeypatch):
    holder = {"payload": None, "boom": False, "text": "（纯文本回退）"}

    def _get_chat_model(*a, **k):
        return _Model(holder["payload"], holder["boom"])

    def _llm_invoke(*a, **k):
        return holder["text"]

    monkeypatch.setattr(ir_mod, "get_chat_model", _get_chat_model)
    monkeypatch.setattr(ir_mod, "llm_invoke", _llm_invoke)
    return holder


def _agent(with_context: bool = True) -> InterviewReviewAgent:
    agent = InterviewReviewAgent()
    agent.extra_context = "【面试记录】..." if with_context else ""
    return agent


# ---------------------------------------------------------------- 元数据
def test_agent_metadata():
    agent = create_agent(MODE_INTERVIEW_REVIEW)
    assert isinstance(agent, InterviewReviewAgent)
    assert not isinstance(agent, JDMatchAgent)
    assert agent.mode == MODE_INTERVIEW_REVIEW
    assert agent.artifact == "Interview_Review"
    assert agent.one_shot is True
    assert agent.requires_confirmation is False


# ---------------------------------------------------------------- 结构化输出
def test_structured_output_is_parsed_and_rendered(patch_llm):
    patch_llm["payload"] = _sample_review()
    agent = _agent()
    text = agent.respond([HumanMessage(content="帮我复盘刚才的面试")])

    assert agent.result is not None and agent.result.overall == 72
    assert "## 面试复盘：72/100" in text
    assert "整体思路清晰，技术细节需要夯实。" in text
    assert "### 分项得分" in text and "- 技术深度：70/100" in text
    assert "### 追问链还原" in text and "1. 自我介绍" in text
    assert "### 薄弱点" in text and "- 对向量检索原理回答含糊" in text
    assert "### 改进动作" in text and "- [ ] 复盘 HNSW 原理" in text


def test_render_omits_empty_sections():
    text = render_interview_review(InterviewReview(overall=0))
    assert "### 分项得分" not in text
    assert "### 追问链还原" not in text
    assert "### 薄弱点" not in text
    assert "### 改进动作" not in text


def test_review_fields_align_with_frontend_contract():
    """字段需覆盖前端复盘视图所需，避免出现第二条重复链路。"""
    review = _sample_review()
    dumped = review.model_dump()
    for field in ("overall", "dimensions", "question_chain", "weaknesses", "suggestions", "detail"):
        assert field in dumped


# ---------------------------------------------------------------- 降级
def test_falls_back_to_text_when_structured_raises(patch_llm):
    patch_llm["boom"] = True
    patch_llm["text"] = "回退的纯文本点评"
    agent = _agent()
    text = agent.respond([HumanMessage(content="x")])
    assert agent.result is None
    assert "回退的纯文本点评" in text


def test_falls_back_when_structured_returns_wrong_type(patch_llm):
    patch_llm["payload"] = 123
    patch_llm["text"] = "回退的纯文本点评"
    agent = _agent()
    assert "回退的纯文本点评" in agent.respond([HumanMessage(content="x")])
    assert agent.result is None


def test_missing_context_prepends_tip(patch_llm):
    patch_llm["payload"] = _sample_review()
    agent = _agent(with_context=False)
    assert agent.respond([HumanMessage(content="x")]).startswith(MISSING_CONTEXT_TIP)


def test_context_present_has_no_tip(patch_llm):
    patch_llm["payload"] = _sample_review()
    agent = _agent(with_context=True)
    assert MISSING_CONTEXT_TIP not in agent.respond([HumanMessage(content="x")])


# ---------------------------------------------------------------- 其他
def test_finalize_prefers_rendered_report():
    agent = _agent()
    agent.result = _sample_review()
    assert agent.finalize("原始转录") == render_interview_review(agent.result)


def test_respond_stream_joins_to_full_text(patch_llm):
    patch_llm["payload"] = _sample_review()
    agent = _agent()
    pieces = list(agent.respond_stream([HumanMessage(content="x")]))
    assert "".join(pieces) and "## 面试复盘：72/100" in "".join(pieces)
