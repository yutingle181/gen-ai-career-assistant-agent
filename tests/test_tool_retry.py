"""工具失败分级与数据时效（P0-2 / P0-3）。

覆盖四件事：
1. 失败分类正确（超时 / 网络 / 5xx / 4xx / 未知）；
2. 分级重试策略：同通道退避重试 → 换参数（简化检索式）→ 换通道；4xx 不在同通道重试；
3. 失败最终「如实降级」：带标记、带分类，且上层（事件流）能识别成 ok=False；
4. 数据时效：抓取时间与文档时间拿得到就注入，拿不到就显式写「未知」。

全部离线：不打真实网络、不依赖 API Key、不真的等待超时。
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from src import config, tools

pytest.importorskip("langchain_community")


@pytest.fixture(autouse=True)
def _fast_and_online(monkeypatch):
    """联网开关打开，但退避归零、超时 1 秒：用例快且确定。"""
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)
    monkeypatch.setattr(config, "WEB_SEARCH_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(config, "WEB_SEARCH_TIMEOUT", 1)
    monkeypatch.setattr(config, "ENABLE_FETCH_TIME", True)


class _Boom(Exception):
    """带 HTTP 状态码的假异常（模拟 SDK 的 APIStatusError）。"""

    def __init__(self, status: int | None = None, msg: str = "boom") -> None:
        super().__init__(msg)
        if status is not None:
            self.status_code = status


def _patch_search(monkeypatch, outcomes: list) -> list[tuple[str, str]]:
    """按序消费 `outcomes`：异常则抛，字符串则返回；返回可断言的调用记录。"""
    calls: list[tuple[str, str]] = []

    def _fake(backend: str, query: str, max_results: int) -> str:
        calls.append((backend, query))
        item = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(tools, "_search_once", _fake)
    return calls


def _stuck_thread(monkeypatch) -> None:
    """让守护线程永远「还活着」→ 走超时分支（不真的等待）。"""

    class _FakeThread:
        def __init__(self, target=None, daemon=None):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

    monkeypatch.setattr(tools.threading, "Thread", _FakeThread)


# ------------------------------------------------------------------ 失败分类
def test_classify_error_reads_status_code_first():
    """结构化信息（状态码）优先，文本匹配兜底。"""
    assert tools.classify_error(_Boom(503)) == "http_5xx"
    assert tools.classify_error(_Boom(400)) == "http_4xx"
    assert tools.classify_error(TimeoutError("slow")) == "timeout"
    assert tools.classify_error(_Boom(msg="Connection reset by peer")) == "network"
    assert tools.classify_error(_Boom(msg="503 Service Unavailable")) == "http_5xx"
    assert tools.classify_error(_Boom()) == "unknown"


# ------------------------------------------------------------------ 分级重试
def test_same_channel_retry_then_success(monkeypatch):
    """瞬态失败先在同通道重试；第二次成功即返回，并带抓取时间。"""
    calls = _patch_search(monkeypatch, [_Boom(msg="Connection reset"), "模拟搜索结果"])
    out = tools.web_search_detailed("长沙 AI 岗位")

    assert out.ok and out.error_kind == ""
    assert out.attempts == 2
    assert "模拟搜索结果" in out.text
    assert "抓取时间" in out.text
    assert len(calls) == 2


def test_http_4xx_is_not_retried_on_same_channel(monkeypatch):
    """4xx（参数 / 权限）重试不会变好：同通道只试一次，直接进入下一策略。"""
    query = '"过长 的 检索 式" ' + "词" * 80
    calls = _patch_search(monkeypatch, [_Boom(400)])
    out = tools.web_search_detailed(query)

    assert not out.ok and out.error_kind == "http_4xx"
    assert out.attempts == len(tools._search_plan(query))
    assert len(calls) == out.attempts


def test_all_failures_degrade_with_kind(monkeypatch):
    """全部尝试失败：返回空文本 + 明确分类，且不抛异常。"""
    _stuck_thread(monkeypatch)
    _patch_search(monkeypatch, ["不会被用到"])
    out = tools.web_search_detailed("任意问题")

    assert out.text == "" and out.error_kind == "timeout" and out.attempts >= 1


def test_search_plan_order_and_alt_backend(monkeypatch):
    """检索计划必须是「原式 → 简化式 → 备选通道」，且策略数量随配置变化。"""
    monkeypatch.setattr(config, "WEB_SEARCH_ALT_BACKEND", "text")
    long_query = '"长沙 AI 应用工程师 岗位" ' + "补" * 80
    plan = tools._search_plan(long_query)

    assert plan[0] == ("auto", long_query)
    assert len(plan) == 3 and plan[-1][0] == "text"
    assert plan[1][1] != long_query  # 中间那条是简化后的检索式

    monkeypatch.setattr(config, "WEB_SEARCH_ALT_BACKEND", "")
    assert len(tools._search_plan("短问题")) == 1


# ------------------------------------------------------------------ 降级契约
def test_degraded_marker_round_trip():
    """降级文本必须能解析回失败分类（事件流与指标都依赖这个契约）。"""
    for kind in ("timeout", "http_5xx", "http_4xx", "empty", "unknown"):
        text = tools.degraded_text(kind)
        assert text.startswith(tools.DEGRADED_MARK)
        assert tools.degraded_kind(text) == kind

    assert tools.degraded_kind("正常检索结果") == ""
    assert "暂不可用" in tools.degraded_text("timeout")  # 保留旧措辞，兼容既有调用方
    assert "未联网核实" in tools.degraded_text("timeout")


def test_search_tool_returns_degraded_text_on_failure(monkeypatch):
    """bind_tools 暴露的工具在失败时回灌可读事实说明，而不是空串或异常。"""
    _stuck_thread(monkeypatch)
    _patch_search(monkeypatch, ["不会被用到"])
    tool = tools.get_agent_tools()[0]

    out = tool.invoke({"query": "任意问题"})
    assert tools.degraded_kind(out) == "timeout"
    assert "暂不可用" in out


def test_tool_loop_marks_soft_failure_as_not_ok(monkeypatch):
    """工具「软失败」在事件流里必须是 ok=False + 带失败分类。

    这条守住的是「可观测」：以前工具返回一句含糊提示也算 ok=True，
    前端时间线与指标都看不出「这轮其实没查到东西」。
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool as lc_tool

    from src.graph import tool_loop

    @lc_tool
    def search_web(query: str) -> str:
        """伪工具：恒定返回降级文本。"""
        return tools.degraded_text("timeout")

    class _FakeModel:
        def __init__(self) -> None:
            self._idx = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages, **kwargs):
            self._idx += 1
            if self._idx == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "c1"}],
                )
            return AIMessage(content="基于已有信息作答")

    monkeypatch.setattr(tool_loop, "get_chat_model", lambda **kwargs: _FakeModel())
    events: list[dict] = []
    out = tool_loop.run_with_tools(
        [HumanMessage(content="查")], tools=[search_web], on_event=events.append
    )

    assert out == "基于已有信息作答"
    assert events and events[0]["ok"] is False and events[0]["error_kind"] == "timeout"


# ------------------------------------------------------------------ 数据时效
def test_fetch_time_can_be_disabled(monkeypatch):
    """时效注入有开关：关掉后文本回到改动前的样子（零回归路径）。"""
    _patch_search(monkeypatch, ["结果"])
    monkeypatch.setattr(config, "ENABLE_FETCH_TIME", False)

    assert "抓取时间" not in tools.web_search_detailed("问题").text


def test_document_time_prefers_meta_then_file_mtime(tmp_path):
    """文档时间提取：先取入库元信息，再按来源文件 mtime 推断，都没有就「未知」。

    注意契约分层：`_document_time_value` 只负责**取到原始时间**，
    `_document_time` 在它之上加「距今天数 / 是否可能过期」的标注
    （标注的测试在 tests/test_freshness.py）。
    """
    from_meta = SimpleNamespace(source="(不存在).md", meta={"updated_at": "2025-01-02 10:30:00"})
    assert tools._document_time_value(from_meta) == "2025-01-02 10:30"

    doc = tmp_path / "kb.md"
    doc.write_text("内容", encoding="utf-8")
    by_mtime = tools._document_time_value(SimpleNamespace(source=str(doc), meta={}))
    assert by_mtime[:4] == str(datetime.now().year)

    assert tools._document_time_value(SimpleNamespace(source="缺失.md", meta={})) == "未知"

    # 标注层：过期资料必须被写成「可能已过期」，而不是原样输出一个日期
    assert "可能已过期" in tools._document_time(from_meta)


def test_knowledge_tool_marks_document_time(monkeypatch):
    """知识库片段必须带文档时间；查不到来源时也要显式写「未知」，不许留空。"""
    chunk = SimpleNamespace(source="文档A.md", page=3, text="片段内容", meta={})
    # 知识库工具与联网检索共用同一套「超时 + 分级」实现（_run_with_status），
    # 因此这里打桩的是它，而不是更底层的 _run_with_timeout。
    monkeypatch.setattr(tools, "_run_with_status", lambda fn, timeout, what: (True, [chunk], ""))

    class _Pipeline:
        def retrieve(self, query, cfg=None):
            return [chunk]

    class _Registry:
        def get(self, name):
            return _Pipeline()

    import src.rag.registry as registry

    monkeypatch.setattr(registry, "get_registry", lambda: _Registry())

    out = tools._knowledge_search_tool("kb").invoke({"query": "问题"})
    assert "来源：文档A.md 第3页" in out
    assert "文档时间：未知" in out
