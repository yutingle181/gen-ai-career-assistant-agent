"""JD 匹配诊断场景：结构化输出解析、降级提示与渲染（全离线，无需 API Key）。

覆盖：
- 结构化输出成功 -> 渲染为评分卡 Markdown 并缓存 result；
- 结构化输出异常/类型异常 -> 回退为纯文本，不空手而归；
- 缺 user_context -> 前置缺口提示；有 user_context -> 不加提示；
- finalize 优先使用渲染好的报告；
- 元数据与工厂注册。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.agents import create_agent
from src.agents import jd_match as jd_mod
from src.agents.jd_match import (
    DIMENSION_LABELS,
    MISSING_CONTEXT_TIP,
    JDMatchAgent,
    render_jd_match,
)
from src.models import JDMatchResult
from src.state import MODE_JD_MATCH


def _sample_result() -> JDMatchResult:
    return JDMatchResult(
        total_score=78,
        dimension_scores={"skills": 80, "experience": 70, "education": 90, "projects": 72},
        matched=["Python", "LangChain"],
        gaps=["Kubernetes"],
        interview_focus=["准备分布式部署案例"],
        summary="整体匹配良好，部署经验是主要短板。",
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
    """把 JD 匹配模块内的模型与文本调用替换为可控桩。"""
    holder = {"payload": None, "boom": False, "text": "（纯文本回退）"}

    def _get_chat_model(*a, **k):
        return _Model(holder["payload"], holder["boom"])

    def _llm_invoke(*a, **k):
        return holder["text"]

    monkeypatch.setattr(jd_mod, "get_chat_model", _get_chat_model)
    monkeypatch.setattr(jd_mod, "llm_invoke", _llm_invoke)
    return holder


def _agent(with_context: bool = True) -> JDMatchAgent:
    agent = JDMatchAgent()
    agent.extra_context = "【岗位】AI 应用工程师\n【简历】..." if with_context else ""
    return agent


# ---------------------------------------------------------------- 元数据
def test_agent_metadata():
    agent = create_agent(MODE_JD_MATCH)
    assert isinstance(agent, JDMatchAgent)
    assert agent.mode == MODE_JD_MATCH
    assert agent.artifact == "JD_Match"
    assert agent.one_shot is True
    assert agent.requires_confirmation is True


# ---------------------------------------------------------------- 结构化输出
def test_structured_output_is_parsed_and_rendered(patch_llm):
    patch_llm["payload"] = _sample_result()
    agent = _agent()
    text = agent.respond([HumanMessage(content="简历和这个 JD 匹配吗")])

    assert agent.result is not None and agent.result.total_score == 78
    assert "## JD 匹配诊断：78/100" in text
    assert "### 分项得分" in text
    assert f"- {DIMENSION_LABELS['skills']}：80/100" in text
    assert "### 命中项" in text and "- Python" in text
    assert "### 缺口项" in text and "- Kubernetes" in text
    assert "### 面试准备重点" in text
    assert agent.citations == []


def test_render_handles_unknown_dimension_key():
    result = JDMatchResult(total_score=60, dimension_scores={"custom": 60})
    text = render_jd_match(result)
    assert "- custom：60/100" in text


def test_render_omits_empty_sections():
    text = render_jd_match(JDMatchResult(total_score=0))
    assert "### 命中项" not in text
    assert "### 缺口项" not in text
    assert "### 面试准备重点" not in text


# ---------------------------------------------------------------- 降级
def test_falls_back_to_text_when_structured_raises(patch_llm):
    patch_llm["payload"] = None
    patch_llm["boom"] = True
    patch_llm["text"] = "回退的纯文本诊断"
    agent = _agent()
    text = agent.respond([HumanMessage(content="x")])

    assert agent.result is None
    assert "回退的纯文本诊断" in text


def test_falls_back_when_structured_returns_wrong_type(patch_llm):
    patch_llm["payload"] = "not a result"
    patch_llm["text"] = "回退的纯文本诊断"
    agent = _agent()
    assert "回退的纯文本诊断" in agent.respond([HumanMessage(content="x")])
    assert agent.result is None


def test_missing_context_prepends_tip(patch_llm):
    patch_llm["payload"] = _sample_result()
    agent = _agent(with_context=False)
    text = agent.respond([HumanMessage(content="简历和这个 JD 匹配吗")])
    assert text.startswith(MISSING_CONTEXT_TIP)


def test_context_present_has_no_tip(patch_llm):
    patch_llm["payload"] = _sample_result()
    agent = _agent(with_context=True)
    assert MISSING_CONTEXT_TIP not in agent.respond([HumanMessage(content="x")])


def test_fallback_text_also_gets_tip(patch_llm):
    patch_llm["boom"] = True
    patch_llm["text"] = "回退文本"
    agent = _agent(with_context=False)
    assert agent.respond([HumanMessage(content="x")]).startswith(MISSING_CONTEXT_TIP)


# ---------------------------------------------------------------- 其他
def test_finalize_prefers_rendered_report():
    agent = _agent()
    agent.result = _sample_result()
    assert agent.finalize("原始 transcript") == render_jd_match(agent.result)


def test_finalize_falls_back_to_transcript_tail():
    agent = _agent()
    assert agent.finalize("只有转录") == "只有转录"


def test_respond_stream_joins_to_full_text(patch_llm):
    patch_llm["payload"] = _sample_result()
    agent = _agent()
    pieces = list(agent.respond_stream([HumanMessage(content="x")]))
    assert pieces
    assert "".join(pieces) == render_jd_match(agent.result)
