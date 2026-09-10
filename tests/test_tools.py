"""工具层：联网搜索降级 + 知识库检索工具。

联网搜索必须「失败即降级」，不能阻塞主流程——这里用假线程确定性覆盖超时分支，
不依赖真实网络。
"""

from __future__ import annotations

import pytest

from src import config, tools


def test_web_search_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", False)
    assert tools.web_search("任意问题") == ""


def test_web_search_timeout_degrades_to_empty(monkeypatch):
    """模拟检索线程超时：必须返回空串降级，而不是一直阻塞。"""
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)

    class _FakeThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True  # 模拟 worker 仍未结束

    monkeypatch.setattr(tools.threading, "Thread", _FakeThread)
    assert tools.web_search("任意问题", timeout=1) == ""


def test_get_agent_tools_returns_list(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", False)
    assert isinstance(tools.get_agent_tools(), list)


def test_knowledge_search_tool_reports_missing_kb():
    pytest.importorskip("langchain_core.tools")
    tool = tools._knowledge_search_tool("__no_such_kb__")
    out = tool.invoke({"query": "任意问题"})
    assert "不存在" in out


class _DDG:
    """确定性 stub：模拟 DuckDuckGo 检索返回。"""

    def __init__(self, num_results: int = 5) -> None:
        self.num_results = num_results

    def run(self, query: str) -> str:
        return "模拟搜索结果内容"


def test_web_search_success(monkeypatch):
    import langchain_community.tools as lct

    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)
    monkeypatch.setattr(lct, "DuckDuckGoSearchResults", lambda num_results=5: _DDG(num_results))
    assert "模拟搜索结果" in tools.web_search("任意问题", timeout=5)


def test_web_search_falls_back_on_exception(monkeypatch):
    import langchain_community.tools as lct

    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)

    class _Boom:
        def __init__(self, num_results: int = 5) -> None:
            self.num_results = num_results

        def run(self, query: str) -> str:
            raise RuntimeError("ddg down")

    monkeypatch.setattr(lct, "DuckDuckGoSearchResults", _Boom)
    assert tools.web_search("任意问题", timeout=5) == ""


def test_get_agent_tools_includes_web_and_kb(monkeypatch):
    import langchain_community.tools as lct

    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)
    monkeypatch.setattr(lct, "DuckDuckGoSearchResults", lambda num_results=5: _DDG(num_results))
    got = tools.get_agent_tools("some_kb")
    assert len(got) == 2  # 联网工具 + 知识库工具


def test_get_agent_tools_web_import_failure_keeps_kb(monkeypatch):
    import langchain_community.tools as lct

    monkeypatch.delattr(lct, "DuckDuckGoSearchResults", raising=True)
    got = tools.get_agent_tools("some_kb")
    assert len(got) == 1  # 仅知识库工具


def test_get_agent_tools_kb_failure_keeps_web(monkeypatch):
    import langchain_community.tools as lct

    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)
    monkeypatch.setattr(lct, "DuckDuckGoSearchResults", lambda num_results=5: _DDG(num_results))
    monkeypatch.setattr(
        tools, "_knowledge_search_tool", lambda name: (_ for _ in ()).throw(RuntimeError("kb boom"))
    )
    got = tools.get_agent_tools("some_kb")
    assert len(got) == 1  # 仅联网工具


def test_knowledge_search_tool_returns_results(monkeypatch):
    pytest.importorskip("langchain_core.tools")
    from src.rag import registry as rag_registry

    class _Chunk:
        def __init__(self, text, source="d1", page=None):
            self.text = text
            self.source = source
            self.page = page

    class _KB:
        def retrieve(self, query):
            return [_Chunk("答案一"), _Chunk("答案二", "d2")]

    class _Reg:
        def get(self, name):
            return _KB()

    monkeypatch.setattr(rag_registry, "get_registry", lambda: _Reg())
    out = tools._knowledge_search_tool("demo").invoke({"query": "问题"})
    assert "答案一" in out and "答案二" in out


def test_knowledge_search_tool_empty_results(monkeypatch):
    pytest.importorskip("langchain_core.tools")
    from src.rag import registry as rag_registry

    class _KB:
        def retrieve(self, query):
            return []

    class _Reg:
        def get(self, name):
            return _KB()

    monkeypatch.setattr(rag_registry, "get_registry", lambda: _Reg())
    out = tools._knowledge_search_tool("demo").invoke({"query": "问题"})
    assert "未检索到" in out

