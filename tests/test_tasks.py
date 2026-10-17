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
from src.tasks_store import TaskStoreRouter
from tests.test_tasks_store import FakeRedis, _ExplodingStore


def _redis_router(monkeypatch) -> TaskStoreRouter:
    """给当前测试装一个"共享后端的假 Redis"路由器（覆盖跨进程分支）。"""
    client = FakeRedis()
    router = TaskStoreRouter()
    from src.tasks_store import RedisTaskStore

    router._active = RedisTaskStore(
        config.TASK_STATE_REDIS_URL, "agent:task:", 600, client=client
    )
    tasks._STORE = router
    return router


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    tasks.reset()
    metrics.reset()
    monkeypatch.setattr(config, "API_MAX_QUEUE", 8)
    monkeypatch.setattr(config, "API_QUEUE_TIMEOUT", 60.0)
    # 钉住任务状态后端为 memory：`.env` 被 gitignore，CI 上没有这个文件，
    # 而本机把 TASK_STATE_BACKEND 设成 redis 就会让"默认是进程内"的用例全红。
    # 隐式依赖默认值的测试是脆弱的：前提必须由测试自己写出来（2026-10-16 实测踩到）。
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "memory")
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


# ---------------------------------------------------------------- 跨副本取消（v1.7.0）
def test_cancel_reaches_task_running_in_other_replica(monkeypatch):
    """**改造前就是这条静默失败**：任务在别的副本上跑，本地没有句柄，
    旧实现直接 return False（→ /chat/cancel 回 status=not_found，用户以为停了、token 照烧）。

    新实现要能在共享状态里找到那个任务并写取消位。
    """
    router = _redis_router(monkeypatch)
    router.save("sess-x", {"session_id": "sess-x", "status": "running"})

    assert tasks.current("sess-x") is None, "本地确实没有句柄（它在另一个副本上）"
    assert tasks.cancel("sess-x", "user") is True, "必须返回 True：取消信号已送达共享状态"
    assert router.cancel_flag("sess-x") == "user"


def test_cancel_returns_false_when_task_absent_everywhere(monkeypatch):
    """共享状态里也查无此会话 → 老老实实返回 False（不能无中生有地"取消成功"）。"""
    _redis_router(monkeypatch)
    assert tasks.cancel("never-existed") is False


def test_cancel_if_running_ignores_finished_task_in_shared_state(monkeypatch):
    """断连兜底：任务已在共享状态里结束，就不该再标成取消（否则 /health 会撒谎）。"""
    router = _redis_router(monkeypatch)
    router.save("sess-done", {"session_id": "sess-done", "status": "done"})
    assert tasks.cancel_if_running("sess-done", "client_gone") is False
    assert router.cancel_flag("sess-done") is None


def test_cancel_if_running_cross_replica(monkeypatch):
    """跨副本的断连兜底：任务在别的副本上 running → 仍要能取消。"""
    router = _redis_router(monkeypatch)
    router.save("sess-r", {"session_id": "sess-r", "status": "running"})
    assert tasks.cancel_if_running("sess-r", "client_gone") is True
    assert router.cancel_flag("sess-r") == "client_gone"


def test_backend_reports_current_store(monkeypatch):
    """后端名可观察：降级是静默的，不给这个字段就只能翻早已滚过去的启动日志。"""
    assert tasks.backend() == "memory"
    _redis_router(monkeypatch)
    assert tasks.backend() == "redis"


def test_prune_clears_shared_state_too(monkeypatch):
    """清理本地句柄时也要清共享状态，否则 Redis 里会留下终态任务直到 TTL 到期。"""
    router = _redis_router(monkeypatch)
    handle = tasks.register("sess-p")
    handle.mark_running()
    handle.finish(tasks.DONE)
    assert tasks.prune() == 1
    assert router.status_of("sess-p") is None


def test_remote_cancel_sets_local_handle_immediately(monkeypatch):
    """收到广播 → 本地句柄**立刻**置位（这正是三保险里"广播"那一路的作用：
    不用等到下一个轮询节流窗口才停）。"""
    _redis_router(monkeypatch)
    handle = tasks.register("sess-b")
    handle.mark_running()
    assert handle.should_stop() is False

    tasks._apply_remote_cancel("sess-b")

    assert handle.should_stop() is True
    assert handle.cancel_reason == "remote"
    assert metrics.counter("task.cancel.remote") == 1


def test_remote_cancel_ignores_unknown_or_finished_session(monkeypatch):
    """广播一个本进程没有的（或已结束的）会话：静默忽略，不能抛给订阅线程。"""
    _redis_router(monkeypatch)
    tasks._apply_remote_cancel("never-heard-of-it")

    handle = tasks.register("sess-c")
    handle.finish(tasks.DONE)
    tasks._apply_remote_cancel("sess-c")
    assert handle.cancelled is False, "已结束的任务不该被重新标成取消"


def test_store_subscribes_only_once(monkeypatch):
    """首次用到存储时订阅一次即可：重复订阅会退化成同一进程收到多次广播。

    这里刻意用**真实的 TaskStoreRouter**（只把后端换成假 Redis 客户端），
    而不是自己写个替身：2026-10-16 真机验证时才发现路由器漏了 subscribe_cancel，
    而当时的单测替身自带同名方法 → 替身与真实对象一起自洽，测试全绿、真机 /health 直接 500。
    """
    from src.tasks_store import RedisTaskStore

    subscribed = []
    real_router = TaskStoreRouter()
    real_router._active = RedisTaskStore("redis://x/0", "agent:task:", 600, client=FakeRedis())
    real_router.subscribe_cancel = lambda cb: subscribed.append(cb)

    monkeypatch.setattr(tasks, "TaskStoreRouter", lambda: real_router)
    tasks.reset_store()

    tasks._store()
    tasks._store()
    assert len(subscribed) == 1
    assert subscribed[0] is tasks._apply_remote_cancel


def test_router_exposes_subscribe_cancel():
    """路由器必须有 subscribe_cancel（tasks.py 是在路由器上发起订阅的）。

    这是回归用例：漏掉它时，`/health` 会在真机上抛 AttributeError 直接 500。
    """
    assert callable(TaskStoreRouter().subscribe_cancel)


def test_cancel_response_status_comes_from_shared_state_when_local_missing(monkeypatch):
    """跨副本取消时 /chat/cancel 的 status 不能报 not_found。

    回归背景（2026-10-16 真机验证发现）：任务跑在另一个副本上时，
    响应是 `{"cancelled": true, "status": "not_found"}` —— 取消已经在对方进程生效了，
    却告诉调用方「没找到任务」，正是这次改造要消灭的那类静默误导。
    """
    from src.api.routers.chat import chat_cancel
    from src.api.schemas import CancelRequest

    router = _redis_router(monkeypatch)
    router.save("sess-remote", {"session_id": "sess-remote", "status": "running"})

    resp = asyncio.run(chat_cancel(CancelRequest(session_id="sess-remote", reason="user")))

    assert resp["cancelled"] is True
    assert resp["status"] == "running", "本地没句柄时应读共享状态，而不是谎称 not_found"


def test_shared_status_survives_backend_failure(monkeypatch):
    """读共享状态失败时返回 None，不能把取消接口一起搞挂。"""
    _redis_router(monkeypatch)
    tasks._STORE._active = _ExplodingStore()
    assert tasks.shared_status("sess-x") is None
