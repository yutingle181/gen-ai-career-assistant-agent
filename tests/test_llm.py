"""LLM 接入层单元测试（离线、确定性）。

通过 monkeypatch `get_chat_model` 为 fake 模型，覆盖：构造前置校验、温度绑定、
重试、流式输出、健康检查、消息拼装等分支，不依赖真实网络或 Key。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import config
from src.llm import (
    build_messages,
    check_llm_health,
    get_chat_model,
    llm_invoke,
    llm_stream,
    reset_chat_model,
)


class _FakeModel:
    def __init__(self) -> None:
        self.bound: float | None = None

    def bind(self, temperature=None):
        self.bound = temperature
        return self

    def invoke(self, msgs):
        return AIMessage(content="回复内容")

    def stream(self, msgs):
        for t in ("你", "好"):
            yield AIMessage(content=t)


def test_build_messages():
    msgs = build_messages("sys", [HumanMessage(content="hi")])
    assert msgs[0].content == "sys"
    assert isinstance(msgs[1], HumanMessage)


def test_get_chat_model_requires_key(monkeypatch):
    reset_chat_model()
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError):
        get_chat_model()


def test_get_chat_model_build_and_bind(monkeypatch):
    import src.llm as llm

    monkeypatch.setattr(config, "OPENAI_API_KEY", "k")
    monkeypatch.setattr(llm, "ChatOpenAI", lambda **kw: _FakeModel())
    reset_chat_model()
    m = get_chat_model()
    assert m is not None
    m2 = get_chat_model(temperature=0.2)
    assert m2.bound == 0.2


def test_llm_invoke_returns_text(monkeypatch):
    import src.llm as llm

    monkeypatch.setattr(llm, "get_chat_model", lambda temperature=None: _FakeModel())
    assert llm_invoke([HumanMessage(content="q")]) == "回复内容"


def test_llm_invoke_list_content(monkeypatch):
    import src.llm as llm

    class _ListModel(_FakeModel):
        def invoke(self, msgs):
            return AIMessage(content=[{"text": "a"}, {"text": "b"}])

    monkeypatch.setattr(llm, "get_chat_model", lambda temperature=None: _ListModel())
    assert llm_invoke([HumanMessage(content="q")]) == "ab"


def test_llm_invoke_retries_on_failure(monkeypatch):
    import src.llm as llm

    class _Boom:
        def __init__(self) -> None:
            self.n = 0

        def invoke(self, msgs):
            self.n += 1
            raise RuntimeError("timeout")

    boom = _Boom()
    monkeypatch.setattr(llm, "get_chat_model", lambda temperature=None: boom)
    with pytest.raises(RuntimeError):
        llm_invoke([HumanMessage(content="q")])
    assert boom.n >= 2  # 触发了指数退避重试


def test_llm_stream_yields_chunks(monkeypatch):
    import src.llm as llm

    monkeypatch.setattr(llm, "get_chat_model", lambda temperature=None: _FakeModel())
    assert "".join(llm_stream([HumanMessage(content="q")])) == "你好"


def test_check_llm_health_ok(monkeypatch):
    import src.llm as llm

    monkeypatch.setattr(llm, "llm_invoke", lambda msgs, temperature=None: "正常")
    ok, msg = check_llm_health()
    assert ok and msg


def test_check_llm_health_fails(monkeypatch):
    import src.llm as llm

    def boom(msgs, temperature=None):
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "llm_invoke", boom)
    ok, msg = check_llm_health()
    assert ok is False and "RuntimeError" in msg
