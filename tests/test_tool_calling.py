"""Function Calling 路径：工具调用子图 + 双路径切换 + 工具层超时一致性。

全部使用伪模型与伪工具，离线确定性，不依赖任何 API Key 与真实网络。
重点守住三件事：
1. 闭环能跑通（单步工具 -> 最终作答）；
2. 安全边界生效（轮次上限强制收束、工具异常包装回灌）；
3. 开关关闭时行为与改造前一致，且工具层的超时降级只有一套实现。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src import config, tools
from src.graph import tool_loop

pytest.importorskip("langgraph.prebuilt")


# ------------------------------------------------------------------ 测试替身
def _tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@tool
def search_web(query: str) -> str:
    """伪联网检索工具（测试用）。"""
    return f"搜索结果：{query}"


@tool
def boom_tool(query: str) -> str:
    """总是抛错的工具（测试用），用于验证异常回灌。"""
    raise RuntimeError("upstream down")


class _FakeModel:
    """按脚本返回消息的伪模型；脚本耗尽后重复最后一条。

    `bind_tools` 返回自身，因此绑定版与朴素版共享同一份脚本游标——
    这正好覆盖「收束轮用朴素模型再问一次」的路径。
    """

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self._idx = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        msg = self.script[min(self._idx, len(self.script) - 1)]
        self._idx += 1
        return msg


def _patch_model(monkeypatch, script: list) -> _FakeModel:
    fake = _FakeModel(script)
    monkeypatch.setattr(tool_loop, "get_chat_model", lambda **kwargs: fake)
    return fake


# ------------------------------------------------------------------ 闭环
def test_run_with_tools_completes_single_round(monkeypatch):
    """模型调用一次工具后给出最终答案，并把过程透出为事件。"""
    _patch_model(monkeypatch, [
        _tool_call("search_web", {"query": "长沙 AI 岗位"}),
        AIMessage(content="最终答案"),
    ])
    events: list[dict] = []
    out = tool_loop.run_with_tools(
        [HumanMessage(content="帮我找岗位")], tools=[search_web], on_event=events.append
    )

    assert out == "最终答案"
    assert len(events) == 1
    assert events[0]["name"] == "search_web"
    assert events[0]["ok"] is True
    assert "搜索结果" in events[0]["result_summary"]
    assert isinstance(events[0]["elapsed_ms"], int)


def test_run_with_tools_without_tools_falls_back(monkeypatch):
    """没有可用工具时退回一次普通调用，不抛错。"""
    monkeypatch.setattr(tool_loop, "llm_invoke", lambda messages, **kwargs: "普通回答")
    assert tool_loop.run_with_tools([HumanMessage(content="你好")], tools=[]) == "普通回答"


def test_run_with_tools_allows_multiple_rounds(monkeypatch):
    """两轮工具调用后作答，事件按序累积。"""
    _patch_model(monkeypatch, [
        _tool_call("search_web", {"query": "第一轮"}, "c1"),
        _tool_call("search_web", {"query": "第二轮"}, "c2"),
        AIMessage(content="综合两轮结果"),
    ])
    events: list[dict] = []
    out = tool_loop.run_with_tools(
        [HumanMessage(content="查")], tools=[search_web], max_steps=5, on_event=events.append
    )

    assert out == "综合两轮结果"
    assert [e["name"] for e in events] == ["search_web", "search_web"]


# ------------------------------------------------------------------ 安全边界
def test_run_with_tools_stops_at_max_steps(monkeypatch):
    """模型反复要求调用工具时，达到轮次上限必须强制收束为自然语言作答。"""
    _patch_model(monkeypatch, [
        _tool_call("search_web", {"query": "a"}, "c1"),
        _tool_call("search_web", {"query": "b"}, "c2"),
        AIMessage(content="收束答案"),
    ])
    events: list[dict] = []
    out = tool_loop.run_with_tools(
        [HumanMessage(content="查")], tools=[search_web], max_steps=2, on_event=events.append
    )

    assert out == "收束答案"
    assert len(events) == 1  # 第二轮已达上限，不再执行工具，直接收束作答


def test_tool_exception_is_wrapped_not_raised(monkeypatch):
    """工具抛异常时包装为简短中文回灌，链路不中断，事件标记失败。"""
    _patch_model(monkeypatch, [
        _tool_call("boom_tool", {"query": "x"}),
        AIMessage(content="我改用已有信息回答"),
    ])
    events: list[dict] = []
    out = tool_loop.run_with_tools(
        [HumanMessage(content="查")], tools=[boom_tool], on_event=events.append
    )

    assert out == "我改用已有信息回答"
    assert events[0]["ok"] is False


def test_on_event_callback_error_is_ignored(monkeypatch):
    """事件回调抛错不能影响主流程（埋点故障不应拖垮对话）。"""
    _patch_model(monkeypatch, [
        _tool_call("search_web", {"query": "x"}),
        AIMessage(content="照常作答"),
    ])

    def _bad_callback(_payload):
        raise RuntimeError("callback boom")

    out = tool_loop.run_with_tools(
        [HumanMessage(content="查")], tools=[search_web], on_event=_bad_callback
    )
    assert out == "照常作答"


def test_stream_with_tools_yields_full_text(monkeypatch):
    """流式外壳分片产出，拼接后与原文本一致。"""
    _patch_model(monkeypatch, [
        _tool_call("search_web", {"query": "x"}),
        AIMessage(content="这是一段足够长的最终回答内容"),
    ])
    pieces = list(
        tool_loop.stream_with_tools(
            [HumanMessage(content="查")], tools=[search_web], chunk_size=5
        )
    )
    assert "".join(pieces) == "这是一段足够长的最终回答内容"
    assert len(pieces) > 1


# ------------------------------------------------------------------ 工具层一致性
class _DDG:
    def __init__(self, num_results: int = 5) -> None:
        self.num_results = num_results

    def run(self, query: str) -> str:
        return "模拟搜索结果内容"


def test_agent_web_tool_reuses_search_timeout_degradation(monkeypatch):
    """端到端一致性：检索超时时，bind_tools 暴露的联网工具必须降级为中文提示。

    这是本次修掉的核心缺陷——此前工具列表里直接放裸 DuckDuckGoSearchResults，
    该路径没有守护线程超时，弱网下会把整轮对话拖住。
    """
    import langchain_community.tools as lct

    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)
    monkeypatch.setattr(lct, "DuckDuckGoSearchResults", lambda num_results=5: _DDG())

    class _FakeThread:
        def __init__(self, target=None, daemon=None):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True  # 模拟 worker 卡住

    monkeypatch.setattr(tools.threading, "Thread", _FakeThread)
    got = tools.get_agent_tools()
    assert got, "应至少返回联网工具"

    out = got[0].invoke({"query": "任意问题"})
    assert "暂不可用" in out  # 降级为可读提示，而不是空串或异常


def test_agent_web_tool_returns_result_on_success(monkeypatch):
    """正常路径：工具返回检索内容。"""
    import langchain_community.tools as lct

    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)
    monkeypatch.setattr(lct, "DuckDuckGoSearchResults", lambda num_results=5: _DDG())
    out = tools.get_agent_tools()[0].invoke({"query": "任意问题"})
    assert "模拟搜索结果" in out


def test_get_agent_tools_can_exclude_web(monkeypatch):
    """知识库场景可只要内部资料工具，避免联网结果污染引用来源。"""
    import langchain_community.tools as lct

    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)
    monkeypatch.setattr(lct, "DuckDuckGoSearchResults", lambda num_results=5: _DDG())
    got = tools.get_agent_tools("some_kb", include_web=False)
    assert len(got) == 1
    assert got[0].name == "knowledge_search"


# ------------------------------------------------------------------ 双路径切换
def test_use_tool_calling_needs_both_switch_and_declaration(monkeypatch):
    """启用工具调用必须同时满足：全局开关打开 + 场景声明 needs_tools。"""
    from src.agents.base import BaseAgent

    agent = BaseAgent()
    monkeypatch.setattr(config, "ENABLE_TOOL_CALLING", False)
    assert agent.use_tool_calling is False

    monkeypatch.setattr(config, "ENABLE_TOOL_CALLING", True)
    assert agent.use_tool_calling is False  # 场景未声明

    agent.needs_tools = True
    assert agent.use_tool_calling is True


def test_jobsearch_skips_prefetch_when_tool_calling_on(monkeypatch):
    """开关打开时不再做前置联网检索（改由模型自主调用），但缺失信息提示保留。"""
    from src.agents.jobsearch import JobSearchAgent

    monkeypatch.setattr(config, "ENABLE_TOOL_CALLING", True)
    called: list[str] = []
    monkeypatch.setattr(tools, "web_search", lambda q, **kw: called.append(q) or "结果")

    ctx = JobSearchAgent().prepare("AI 应用工程师 岗位")
    assert called == []
    assert "联网检索结果" not in ctx
    assert "目标城市" in ctx  # MISSING_INFO_TIP 仍在


def test_jobsearch_keeps_prefetch_when_tool_calling_off(monkeypatch):
    """开关关闭（默认）时行为与改造前一致：仍走前置联网检索。"""
    from src.agents.jobsearch import JobSearchAgent

    monkeypatch.setattr(config, "ENABLE_TOOL_CALLING", False)
    monkeypatch.setattr(tools, "web_search", lambda q, **kw: "模拟结果")

    ctx = JobSearchAgent().prepare("长沙 AI 岗位")
    assert "联网检索结果" in ctx
