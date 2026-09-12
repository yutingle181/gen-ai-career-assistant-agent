"""控制生成长度（9.2.A）测试（确定性、无真实网络）。

覆盖：
- ENABLE_MAX_TOKENS / MAX_TOKENS / MAX_TOKENS_BY_MODE 配置默认值正确；
- get_chat_model 按 (模型名, 温度, 流式, max_tokens) 分别缓存，且 None 不向 SDK 透传；
- agents/base.respond / respond_stream 把 max_tokens 透传给 llm 调用；
- SessionManager 主链路按 mode 取上限并随 model 一并透传给 llm。
"""

from __future__ import annotations

import pytest

from src import config
from src.llm import get_chat_model, reset_chat_model


@pytest.fixture
def _cached(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(config, "MODEL_NAME", "qwen-plus")
    monkeypatch.setattr(config, "ENABLE_MODEL_ROUTING", False)
    reset_chat_model()
    yield
    reset_chat_model()


def test_config_defaults():
    assert config.ENABLE_MAX_TOKENS is True
    assert config.MAX_TOKENS == 2048
    # 聊天答疑类收紧、长产物类放宽
    assert config.MAX_TOKENS_BY_MODE["qa"] <= config.MAX_TOKENS_BY_MODE["resume"]
    assert config.MAX_TOKENS_BY_MODE["tutorial"] == config.MAX_TOKENS_BY_MODE["qa"]
    assert "job_search" in config.MAX_TOKENS_BY_MODE


def test_get_chat_model_caches_per_max_tokens(_cached):
    a = get_chat_model(model="qwen-plus", max_tokens=1500)
    b = get_chat_model(model="qwen-plus", max_tokens=1500)
    c = get_chat_model(model="qwen-plus", max_tokens=3500)
    none1 = get_chat_model(model="qwen-plus")  # max_tokens=None
    none2 = get_chat_model(model="qwen-plus")
    assert a is b                       # 同模型同上限复用同一客户端
    assert a is not c                   # 不同上限独立缓存
    assert none1 is none2               # None 上限也复用（键含 None）
    # None 不向 SDK 透传：构造出的客户端 max_tokens 为默认 None
    assert none1.max_tokens is None
    # 给定上限正确写入客户端
    assert a.max_tokens == 1500
    assert c.max_tokens == 3500


def test_base_respond_forwards_max_tokens(monkeypatch):
    import src.agents.base as base

    captured: dict = {}

    def _fake_invoke(messages, model=None, temperature=None, max_tokens=None):
        captured["max_tokens"] = max_tokens
        return "ok"

    monkeypatch.setattr(base, "llm_invoke", _fake_invoke)
    agent = base.BaseAgent()
    agent.respond([], max_tokens=1500)
    assert captured["max_tokens"] == 1500
    agent.respond([])  # 默认 None
    assert captured["max_tokens"] is None


def test_base_respond_stream_forwards_max_tokens(monkeypatch):
    import src.llm as llm_mod

    captured: dict = {}

    def _fake_stream(messages, model=None, temperature=None, max_tokens=None):
        captured["max_tokens"] = max_tokens
        yield "ok"

    # respond_stream 方法内局部 import llm_stream，故需 patch src.llm 上的符号
    monkeypatch.setattr(llm_mod, "llm_stream", _fake_stream)
    from src.agents.base import BaseAgent

    agent = BaseAgent()
    list(agent.respond_stream([], max_tokens=3000))
    assert captured["max_tokens"] == 3000


def test_session_selects_max_tokens_per_mode(monkeypatch):
    import src.agents.base as base
    from src.session import SessionManager
    from src.state import MODE_QA, MODE_RESUME

    monkeypatch.setattr(config, "ENABLE_MAX_TOKENS", True)
    monkeypatch.setattr(config, "ENABLE_MODEL_ROUTING", False)
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", False)

    captured: dict = {}

    def _fake_invoke(messages, model=None, temperature=None, max_tokens=None):
        captured.setdefault("calls", []).append(max_tokens)
        return "（固定回复）"

    monkeypatch.setattr(base, "llm_invoke", _fake_invoke)

    # qa：收紧到 1500
    mgr_qa = SessionManager(MODE_QA)
    mgr_qa.start("你好")
    assert captured["calls"][0] == config.MAX_TOKENS_BY_MODE["qa"]

    # resume：放宽到 3500
    mgr_cv = SessionManager(MODE_RESUME)
    mgr_cv.start("帮我写简历")
    assert captured["calls"][-1] == config.MAX_TOKENS_BY_MODE["resume"]


def test_session_disabled_uses_none(monkeypatch):
    import src.agents.base as base
    from src.session import SessionManager
    from src.state import MODE_QA

    monkeypatch.setattr(config, "ENABLE_MAX_TOKENS", False)
    monkeypatch.setattr(config, "ENABLE_MODEL_ROUTING", False)
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", False)

    captured: dict = {}

    def _fake_invoke(messages, model=None, temperature=None, max_tokens=None):
        captured["max_tokens"] = max_tokens
        return "（固定回复）"

    monkeypatch.setattr(base, "llm_invoke", _fake_invoke)
    mgr = SessionManager(MODE_QA)
    mgr.start("你好")
    assert captured["max_tokens"] is None  # 关闭开关 -> 不传上限，零回归
