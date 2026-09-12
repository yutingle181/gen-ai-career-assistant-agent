"""分级模型路由（9.2.A）测试（确定性、无真实网络）。

覆盖：
- 开关关闭 / 非 qwen 部署 / 带 user_context / 复杂或超长 query / 非简单 mode 均回落 MODEL_STRONG；
- 简单短 query 在 tutorial/qa 模式且无 user_context 时降级到 MODEL_FAST；
- get_chat_model 按 (模型名, 温度, 流式) 分别缓存；
- SessionManager 主链路把路由结果透传给 llm_invoke；
- agents/base.respond / respond_stream 把 model 透传给 llm 调用。
"""

from __future__ import annotations

import pytest

from src import config
from src.llm import get_chat_model, reset_chat_model
from src.model_routing import select_model


@pytest.fixture
def _qwen(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_MODEL_ROUTING", True)
    monkeypatch.setattr(config, "MODEL_STRONG", "qwen-plus")
    monkeypatch.setattr(config, "MODEL_FAST", "qwen-turbo")
    monkeypatch.setattr(config, "ROUTING_SIMPLE_MODES", ("tutorial", "qa"))
    monkeypatch.setattr(config, "ROUTING_SIMPLE_MAX_CHARS", 80)


def test_disabled_returns_strong(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_MODEL_ROUTING", False)
    monkeypatch.setattr(config, "MODEL_STRONG", "qwen-plus")
    assert select_model("你好", mode="qa") == "qwen-plus"


def test_non_qwen_returns_strong(_qwen, monkeypatch):
    # 仅当强模型为 qwen 家族才路由；deepseek-chat 默认部署不路由，零回归
    monkeypatch.setattr(config, "MODEL_STRONG", "deepseek-chat")
    assert select_model("你好", mode="qa") == "deepseek-chat"


def test_user_context_forces_strong(_qwen):
    # 带 JD+简历重任务：即使简单问候也走强模型，保证质量
    assert select_model("你好", mode="qa", user_context="岗位JD与简历") == "qwen-plus"


def test_simple_query_downgrades(_qwen):
    assert select_model("谢谢", mode="qa") == "qwen-turbo"
    assert select_model("你好", mode="tutorial") == "qwen-turbo"
    # 短且无复杂意图、无简单模式命中：保守按简单降级，体现路由收益
    assert select_model("什么是KPI？", mode="qa") == "qwen-turbo"


def test_complex_keyword_stays_strong(_qwen):
    assert select_model("帮我写一段自我介绍", mode="qa") == "qwen-plus"
    assert select_model("帮我分析一下缺点", mode="tutorial") == "qwen-plus"


def test_long_query_stays_strong(_qwen):
    long_q = "这家公司的" * 30  # 150 字符，超过阈值且无复杂关键词
    assert len(long_q) > 80
    assert select_model(long_q, mode="qa") == "qwen-plus"


def test_non_simple_mode_stays_strong(_qwen):
    assert select_model("谢谢", mode="resume") == "qwen-plus"
    assert select_model("你好", mode="interview_questions") == "qwen-plus"


def test_get_chat_model_caches_per_model(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(config, "MODEL_NAME", "qwen-plus")
    reset_chat_model()
    a = get_chat_model(model="qwen-plus")
    b = get_chat_model(model="qwen-plus")
    c = get_chat_model(model="qwen-turbo")
    assert a is b            # 同模型复用同一客户端
    assert a is not c        # 不同模型独立缓存
    d = get_chat_model(model="qwen-plus", temperature=0)
    assert a is not d        # 不同温度也独立缓存
    reset_chat_model()


def test_base_respond_forwards_model(monkeypatch):
    import src.agents.base as base

    captured: dict = {}

    def _fake_invoke(messages, model=None, temperature=None, max_tokens=None):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(base, "llm_invoke", _fake_invoke)
    agent = base.BaseAgent()
    agent.respond([], model="qwen-turbo")
    assert captured["model"] == "qwen-turbo"
    agent.respond([])  # 默认 None
    assert captured["model"] is None


def test_base_respond_stream_forwards_model(monkeypatch):
    import src.llm as llm_mod

    captured: dict = {}

    def _fake_stream(messages, model=None, temperature=None, max_tokens=None):
        captured["model"] = model
        yield "ok"

    # respond_stream 方法内局部 import llm_stream，故需 patch src.llm 上的符号
    monkeypatch.setattr(llm_mod, "llm_stream", _fake_stream)
    from src.agents.base import BaseAgent

    agent = BaseAgent()
    list(agent.respond_stream([], model="qwen-turbo"))
    assert captured["model"] == "qwen-turbo"


def test_session_routes_simple_to_fast(_qwen, monkeypatch):
    import src.agents.base as base
    from src.session import SessionManager
    from src.state import MODE_TUTORIAL

    captured: dict = {}

    def _fake_invoke(messages, model=None, temperature=None, max_tokens=None):
        captured.setdefault("models", []).append(model)
        return "（固定回复）"

    monkeypatch.setattr(base, "llm_invoke", _fake_invoke)
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", False)

    mgr = SessionManager(MODE_TUTORIAL)
    mgr.start("你好")
    assert captured["models"][0] == "qwen-turbo"
