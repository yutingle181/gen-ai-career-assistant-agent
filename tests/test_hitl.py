"""人机协同（Human-in-the-Loop）：草稿态 → 人工确认 → 定稿。

覆盖：Agent 打标、高风险话题识别、SessionManager 状态机（含 one_shot 自动草稿）、
总开关、API 确认端点、产物列表的草稿标注。全部离线可跑。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import config
from src.agents import create_agent
from src.api.main import create_app
from src.safety import check_high_risk
from src.session import SessionManager
from src.state import (
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


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """用固定回复替代 LLM 并关闭联网检索，保证用例快速且确定。"""
    import src.agents.base as base

    monkeypatch.setattr(base, "llm_invoke", lambda *a, **k: "（测试用固定回复）")
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", False)


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    """把产物目录重定向到临时目录，避免污染真实 Agent_output/。"""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------- 场景打标
@pytest.mark.parametrize(
    "mode", [MODE_RESUME, MODE_MOCK_INTERVIEW, MODE_JOB_SEARCH, MODE_JD_MATCH]
)
def test_high_risk_agents_require_confirmation(mode):
    assert create_agent(mode).requires_confirmation is True


@pytest.mark.parametrize(
    "mode",
    [MODE_TUTORIAL, MODE_QA, MODE_INTERVIEW_QUESTIONS, MODE_KNOWLEDGE, MODE_INTERVIEW_REVIEW],
)
def test_normal_agents_do_not_require_confirmation(mode):
    """内容生成类场景不打标，避免确认流程影响演示流畅度。"""
    assert create_agent(mode).requires_confirmation is False


# ---------------------------------------------------------------- 高风险话题
@pytest.mark.parametrize(
    "text",
    [
        "帮我准备薪资谈判，期望薪资怎么谈",
        "公司裁员不给赔偿金怎么办",
        "背调一般会查什么",
        "我的身份证号要填进去吗",
    ],
)
def test_check_high_risk_flags_sensitive_topics(text):
    hit, topic = check_high_risk(text)
    assert hit is True
    assert topic


def test_check_high_risk_passes_normal_text():
    assert check_high_risk("帮我写一篇 LangGraph 教程") == (False, "")


def test_check_high_risk_handles_empty():
    assert check_high_risk("") == (False, "")


# ---------------------------------------------------------------- 状态机
def test_multiturn_session_finishes_as_draft_then_confirm(out_dir):
    mgr = SessionManager(MODE_RESUME)
    mgr.start("帮我写一份简历")
    assert mgr.requires_confirmation is True

    draft = mgr.finish()
    assert mgr.awaiting_confirmation is True
    assert mgr.finished is False, "草稿态不能被当成已定稿"
    assert mgr.artifact_path is None, "artifact_path 语义保留给定稿"
    assert draft and Path(draft).exists()
    assert "_draft" in Path(draft).name

    assert mgr.finish() == draft, "重复 finish 应幂等，不重复落草稿"

    final = mgr.confirm()
    assert mgr.finished is True
    assert mgr.awaiting_confirmation is False
    assert final and Path(final).exists()
    assert "_draft" not in Path(final).name
    assert mgr.confirm() == final, "重复 confirm 应幂等"


def test_one_shot_session_auto_produces_draft(out_dir):
    """one_shot 场景（职位搜索）首轮自动收尾，但只应产出草稿。"""
    mgr = SessionManager(MODE_JOB_SEARCH)
    mgr.start("长沙有哪些 AI 应用工程师岗位")
    assert mgr.awaiting_confirmation is True
    assert mgr.finished is False
    assert mgr.draft_path and Path(mgr.draft_path).exists()


def test_switch_off_finalizes_directly(out_dir, monkeypatch):
    """总开关关闭时恢复为全自动直接定稿。"""
    monkeypatch.setattr(config, "ENABLE_HUMAN_CONFIRM", False)
    mgr = SessionManager(MODE_RESUME)
    mgr.start("帮我写一份简历")
    path = mgr.finish()
    assert mgr.finished is True
    assert mgr.awaiting_confirmation is False
    assert path and "_draft" not in Path(path).name


def test_high_risk_topic_triggers_draft_for_normal_mode(out_dir):
    """非高风险场景一旦命中高风险话题，也应转入待确认。"""
    mgr = SessionManager(MODE_QA)
    mgr.start("公司裁员不给赔偿金，我该怎么维权")
    assert mgr.high_risk_topic
    assert mgr.requires_confirmation is True
    mgr.finish()
    assert mgr.awaiting_confirmation is True


# ---------------------------------------------------------------- 产物列表
def test_list_outputs_labels_draft(out_dir):
    from src.storage import DRAFT_SUFFIX, list_outputs, save_file

    save_file("草稿内容", f"Resume{DRAFT_SUFFIX}")
    save_file("定稿内容", "Resume")
    labels = {i["type"]: i["label"] for i in list_outputs()}
    assert labels[f"Resume{DRAFT_SUFFIX}"] == "简历（草稿）"
    assert labels["Resume"] == "简历"


# ---------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client():
    """关闭鉴权，避免本地 .env 里的 API_KEY 影响用例。"""
    saved = config.API_KEY
    config.API_KEY = ""
    with TestClient(create_app()) as c:
        yield c
    config.API_KEY = saved


def test_chat_then_confirm_flow(client):
    r = client.post(
        "/chat", json={"query": "长沙有哪些 AI 应用工程师岗位", "mode": "job_search"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requires_confirmation"] is True
    assert body["awaiting_confirmation"] is True
    assert body["draft_artifact"], "应返回草稿路径"
    assert body["finished"] is False
    assert body["artifact"] is None

    rc = client.post("/chat/confirm", json={"session_id": body["session_id"]})
    assert rc.status_code == 200, rc.text
    confirmed = rc.json()
    assert confirmed["finished"] is True
    assert confirmed["artifact"], "确认后应落盘定稿"
    assert confirmed["awaiting_confirmation"] is False


def test_confirm_unknown_session_returns_404(client):
    assert client.post("/chat/confirm", json={"session_id": "nope"}).status_code == 404
