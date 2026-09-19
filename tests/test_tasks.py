"""任务生命周期：排队有界、取消真的停手、断连不留空转。

为什么单独一个文件：这三件事都属于「出问题才被注意到」的能力 ——
排队无界会拖垮所有人，取消无效会白烧用户额度，断连不停会一直烧，
而它们在 happy path 上全都看不出来。
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from src import config, metrics, tasks
from src.api.main import create_app


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    tasks.reset()
    metrics.reset()
    monkeypatch.setattr(config, "API_MAX_QUEUE", 8)
    monkeypatch.setattr(config, "API_QUEUE_TIMEOUT", 60.0)
    yield
    tasks.reset()
    metrics.reset()


@pytest.fixture(scope="module")
def client():
    """全局关闭鉴权与 IP 限流，避免本地 .env 影响背压用例。"""
    saved_key, saved_limit = config.API_KEY, config.API_RATE_LIMIT
    config.API_KEY, config.API_RATE_LIMIT = "", 0
    with TestClient(create_app()) as c:
        yield c
    config.API_KEY, config.API_RATE_LIMIT = saved_key, saved_limit


# ---------------------------------------------------------------- 注册表
def test_register_then_running_records_wait():
    handle = tasks.register("s1")
    assert handle.status == tasks.QUEUED

    wait = handle.mark_running()

    assert handle.status == tasks.RUNNING
    assert wait >= 0 and handle.wait_ms == wait


def test_cancel_sets_signal_and_reason():
    handle = tasks.register("s1")
    handle.mark_running()
    assert handle.should_stop() is False

    assert tasks.cancel("s1", reason="user") is True

    assert handle.should_stop() is True
    assert handle.cancel_reason == "user"
    assert metrics.counter("task.cancel.requested") == 1


def test_cancel_on_finished_task_is_noop():
    """已完成的轮次不能被标成取消——否则统计里「取消率」会虚高。"""
    handle = tasks.register("s1")
    handle.mark_running()
    handle.finish(tasks.DONE)

    assert tasks.cancel("s1") is False
    assert handle.cancelled is False


def test_cancel_unknown_session_is_honest():
    assert tasks.cancel("nope") is False
    assert tasks.current("nope") is None


def test_cancel_if_running_skips_queued_tasks():
    handle = tasks.register("s1")  # 还在排队

    assert tasks.cancel_if_running("s1", "client_gone") is False

    handle.mark_running()
    assert tasks.cancel_if_running("s1", "client_gone") is True
    assert metrics.counter("task.cancel.client_gone") == 1


def test_register_supersedes_previous_handle():
    """同一会话重复登记：旧句柄必须被取消，否则 cancel 会打到过期对象上。"""
    first = tasks.register("s1")
    first.mark_running()

    second = tasks.register("s1")

    assert first.cancelled is True and first.cancel_reason == "superseded"
    assert tasks.current("s1") is second


def test_counts_snapshot_and_prune():
    running = tasks.register("a")
    running.mark_running()
    tasks.register("b")  # 排队中

    assert tasks.counts() == {"queued": 1, "running": 1, "total": 2}

    running.finish(tasks.DONE)
    assert tasks.prune() == 1
    snapshot = tasks.snapshot()
    assert snapshot["counts"]["running"] == 0
    assert all(item["status"] != tasks.DONE for item in snapshot["active"])


def test_finish_keeps_first_terminal_state():
    handle = tasks.register("s1")
    handle.mark_running()
    handle.finish(tasks.CANCELLED)
    handle.finish(tasks.DONE)
    assert handle.status == tasks.CANCELLED


# ---------------------------------------------------------------- 有界排队（背压）
def test_acquire_slot_rejects_when_queue_full(monkeypatch):
    """队列满了要立刻回 429，而不是让请求无限期挂着。"""
    monkeypatch.setattr(config, "API_MAX_QUEUE", 0)
    from src.api.routers.chat import _acquire_slot

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_acquire_slot("s1"))

    assert excinfo.value.status_code == 429
    assert "Retry-After" in (excinfo.value.headers or {})
    assert metrics.counter("api.queue.rejected") == 1


def test_acquire_slot_times_out_instead_of_hanging(monkeypatch):
    """排队超时给 503 明确结论，而不是一直等下去。"""
    import src.api.routers.chat as chat

    monkeypatch.setattr(config, "API_QUEUE_TIMEOUT", 1.0)
    sem = asyncio.Semaphore(1)

    async def _run():
        await sem.acquire()  # 占满唯一槽位
        monkeypatch.setattr(chat, "get_semaphore", lambda: sem)
        try:
            await chat._acquire_slot("s1")
        except HTTPException as exc:
            return exc
        return None

    exc = asyncio.run(_run())

    assert exc is not None and exc.status_code == 503
    assert metrics.counter("api.queue.timeout") == 1


def test_acquire_and_release_slot_returns_capacity(monkeypatch):
    import src.api.routers.chat as chat

    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(chat, "get_semaphore", lambda: sem)

    async def _run():
        handle = await chat._acquire_slot("s1")
        assert handle.status == tasks.RUNNING
        chat._release_slot(handle)
        return handle

    handle = asyncio.run(_run())

    assert handle.status == tasks.DONE
    assert sem._value == 1, "槽位必须归还，否则并发额度会被漏掉"
    # 等待时长以分布形式记录（排队慢要能被观测，而不是靠用户抱怨）
    assert "api.queue.wait_ms" in metrics.snapshot()["distributions"]


def test_release_slot_on_disconnect_cancels_running_task(monkeypatch):
    """客户端断开（生成器被关闭）时任务还在跑：必须发取消信号，否则后台照烧额度。"""
    import src.api.routers.chat as chat

    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(chat, "get_semaphore", lambda: sem)

    async def _run():
        handle = await chat._acquire_slot("s1")
        chat._release_slot(handle, disconnected=True)
        return handle

    handle = asyncio.run(_run())

    assert handle.cancelled is True and handle.cancel_reason == "client_gone"
    assert handle.status == tasks.CANCELLED
    assert metrics.counter("task.cancelled") == 1


# ---------------------------------------------------------------- 取消要真的停手
class _AlwaysToolModel:
    """总是要求调用工具的假模型：用来验证「取消后不再继续调工具」。"""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[{"name": "probe_tool", "args": {"x": 1}, "id": "call-1"}],
        )


def _probe_tool(calls: list):
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def probe_tool(x: int) -> str:
        """探测工具。"""
        calls.append(x)
        return "工具结果"

    return probe_tool


def _patch_model(monkeypatch, tool_loop) -> None:
    monkeypatch.setattr(
        tool_loop, "get_chat_model", lambda model=None, max_tokens=None: _AlwaysToolModel()
    )
    monkeypatch.setattr(tool_loop, "llm_invoke", lambda *args, **kwargs: "兜底回答")


def test_tool_loop_executes_tools_without_cancel(monkeypatch):
    """对照组：没有取消信号时工具照常执行（证明后面的用例不是空测）。"""
    from src.graph import tool_loop

    calls: list = []
    _patch_model(monkeypatch, tool_loop)

    tool_loop.run_with_tools(
        [HumanMessage(content="问题")], tools=[_probe_tool(calls)], max_steps=2
    )

    assert calls, "未取消时工具应被执行"


def test_run_with_tools_stops_before_tool_call_when_cancelled(monkeypatch):
    """取消检查点在「要调工具之前」：用户走了就不该再花他的额度继续调工具。"""
    from src.graph import tool_loop

    calls: list = []
    _patch_model(monkeypatch, tool_loop)

    out = tool_loop.run_with_tools(
        [HumanMessage(content="问题")],
        tools=[_probe_tool(calls)],
        max_steps=2,
        should_stop=lambda: True,
    )

    assert calls == [], "取消后不应再执行工具"
    assert out == ""
    assert metrics.counter("agent.cancelled") == 1


def test_stream_with_tools_stops_pushing_after_cancel(monkeypatch):
    """取消后不再往外推字：否则用户会看到「点了停止还在输出」。"""
    from src.graph import tool_loop

    monkeypatch.setattr(tool_loop, "run_with_tools", lambda *a, **k: "很长的一段回答" * 10)

    pieces = list(
        tool_loop.stream_with_tools(
            [HumanMessage(content="问题")], tools=[object()], should_stop=lambda: True
        )
    )

    assert pieces == []


# ---------------------------------------------------------------- 接口契约
def test_cancel_endpoint_reports_signal_sent(client):
    handle = tasks.register("s-live")
    handle.mark_running()

    r = client.post("/chat/cancel", json={"session_id": "s-live"})

    assert r.status_code == 200
    body = r.json()
    assert body["cancelled"] is True
    assert body["status"] == tasks.RUNNING
    assert handle.should_stop() is True
    # 语义必须说清：信号已发出 ≠ 已经停下，调用方不能据此假设资源已释放
    assert "检查点" in body["detail"]


def test_cancel_endpoint_without_task_is_honest(client):
    r = client.post("/chat/cancel", json={"session_id": "no-such-session"})

    assert r.status_code == 200
    assert r.json()["cancelled"] is False
    assert r.json()["status"] == "not_found"


def test_stream_rejected_when_queue_full(client, monkeypatch):
    monkeypatch.setattr(config, "API_MAX_QUEUE", 0)

    r = client.post("/chat/stream", json={"query": "你好"})

    assert r.status_code == 429
    assert r.headers.get("Retry-After")


def test_health_exposes_tasks_and_queue_size(client):
    handle = tasks.register("s-health")
    handle.mark_running()

    r = client.get("/health")

    assert r.status_code == 200
    runtime = r.json()["runtime"]
    assert runtime["tasks"]["counts"]["running"] >= 1
    assert runtime["api_max_queue"] == config.API_MAX_QUEUE


# ---------------------------------------------------------------- 端到端（SSE 真停）
def _fake_manager(sid: str, produced: dict, cancel_after: int | None = None):
    """假的会话管理器：逐片产出，可选在第 N 片后自己发起取消（模拟用户点停止）。"""

    class _FakeManager:
        mode = "qa"
        citations: list = []
        tool_events: list = []
        artifact_path = None
        draft_path = None
        finished = False
        requires_confirmation = False
        awaiting_confirmation = False

        def __init__(self) -> None:
            from types import SimpleNamespace

            self.agent = SimpleNamespace(
                tool_event_listener=None, should_stop=None, one_shot=False
            )

        def step_stream(self, query, auto_finish=False):
            try:
                for i in range(5):
                    produced["n"] += 1
                    if cancel_after is not None and i == cancel_after:
                        tasks.cancel(sid, reason="user")
                    yield f"片段{i}"
            finally:
                # 生成器被 close() 时才会走到这里：这就是「真正停下」的证据
                produced["closed"] = True

    return _FakeManager()


def test_stream_cancel_midway_closes_generator(client, monkeypatch):
    """端到端：中途取消后生成器被关闭、提前停止，并且回执里给出了 cancelled 事件。

    只停止「往外推」是不够的 —— 那等于「前端不显示了，后台照烧额度」。
    """
    import src.api.routers.chat as chat

    produced = {"n": 0, "closed": False}
    sid = "s-stream-cancel"
    manager = _fake_manager(sid, produced, cancel_after=1)
    monkeypatch.setattr(chat, "_ensure_session", lambda req: (sid, manager, False))

    r = client.post("/chat/stream", json={"query": "你好"})
    body = r.text

    assert r.status_code == 200
    assert produced["closed"] is True, "取消后生成器必须被关闭"
    assert produced["n"] < 5, "取消后不应继续生成剩余的片段"
    assert '"type": "cancelled"' in body
    assert '"cancelled": true' in body


def test_stream_skips_generation_when_cancelled_before_start(client, monkeypatch):
    """排队期间就被取消：不该进模型（省下的就是用户的额度）。"""
    import src.api.routers.chat as chat

    produced = {"n": 0, "closed": False}
    sid = "s-queued-cancel"
    manager = _fake_manager(sid, produced)
    handle = tasks.register(sid)
    handle.request_cancel("user")

    async def _acquire(_sid):
        return handle

    monkeypatch.setattr(chat, "_ensure_session", lambda req: (sid, manager, False))
    monkeypatch.setattr(chat, "_acquire_slot", _acquire)

    r = client.post("/chat/stream", json={"query": "你好"})
    body = r.text

    assert r.status_code == 200
    assert produced["n"] == 0, "排队期间取消不应产生任何模型输出"
    assert '"type": "cancelled"' in body
