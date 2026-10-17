"""任务生命周期：排队、在途、取消。

此前一轮对话只有「开始」和「结束」两个状态，于是有两个真实问题：

1. **排队无界无反馈**：`await sem.acquire()` 会一直挂着，客户端只看到「卡住」，
   既不知道在排队、也不知道要等多久，更没法被拒绝；
2. **没有取消**：客户端断开（关标签页 / 断网）后，后台线程继续把这一轮跑完 ——
   用户看不到结果，token 却照烧（而这是用户付费的额度）。

这里把「一个正在跑的任务」显式建模：可查状态、可取消、可统计、可背压。

**跨进程边界（2026-10-16 起已支持，此前是硬限制）**：
状态默认只在进程内（`TASK_STATE_BACKEND=memory`），单副本部署行为与历史版本逐字节一致；
设 `TASK_STATE_BACKEND=redis` 后，「在途状态 + 取消信号 + 背压计数」跨副本共享，
取消请求打到**任意**副本都能生效。否则另一个副本会返回 `not_found`，
取消**静默失败**（用户以为停了，token 照烧）。连不上时自动降级回进程内并告警，
不会阻断服务；当前后端可从 `/health.runtime.task_state_backend` 观察。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from . import metrics
from .logging_setup import get_logger
from .tasks_store import TaskStoreRouter

logger = get_logger(__name__)

#: 任务状态
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"
FAILED = "failed"

_TERMINAL = frozenset({DONE, CANCELLED, FAILED})

#: `should_stop()` 兜底轮询的最小间隔（秒）：流式分片之间也会调它，
#: 不能每个分片都打一次 Redis，但也不能太久（取消要及时）。
_POLL_INTERVAL_SECONDS = 0.2

#: 进程内存储路由器（默认 memory；TASK_STATE_BACKEND=redis 时跨副本共享）
_STORE: TaskStoreRouter | None = None
_STORE_LOCK = threading.Lock()
_SUBSCRIBED = False


@dataclass
class TaskHandle:
    """一个在途任务的句柄：状态 + 取消信号。

    `should_stop()` 既是给**图执行层**（工具轮次之间的检查点）用的，
    也是给**流式网关**（逐段推送时）用的——两边共用一个信号，
    才能做到「用户点了停止，模型/工具真的停下来」，而不是只停掉前端渲染。
    """

    session_id: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = QUEUED
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    wait_ms: int = 0
    cancel_reason: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    #: 上次兜底轮询共享取消位的时刻（节流用，见 should_stop）
    _last_poll: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------ 状态
    def mark_running(self) -> int:
        """从排队进入执行，返回排队等待时长（毫秒）。"""
        self.started_at = time.monotonic()
        self.wait_ms = int((self.started_at - self.queued_at) * 1000)
        self.status = RUNNING
        _mirror(self)  # 同步到存储：redis 模式下别的副本要能看到"开始跑了"
        return self.wait_ms

    def finish(self, status: str = DONE) -> None:
        """结束任务（已结束的任务再调用不会覆盖终态）。"""
        if self.status not in _TERMINAL:
            self.status = status
            _mirror(self)

    @property
    def finished(self) -> bool:
        return self.status in _TERMINAL

    # ------------------------------------------------------------ 取消
    def request_cancel(self, reason: str = "user") -> bool:
        """请求取消；任务已结束则返回 False（避免把「已完成的轮次」标成取消）。"""
        if self.finished:
            return False
        self.cancel_reason = reason or "user"
        self.cancel_event.set()
        return True

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def should_stop(self) -> bool:
        """协作式停止信号：图执行层与流式推送都读它。

        跨进程语义：本地 Event 优先（Pub/Sub 广播会立即置位 ✓）；
        没置位时按 `_POLL_INTERVAL_SECONDS` 节流读一次**共享取消位**兜底 ——
        广播可能丢（订阅者不在线 / 连接中断），轮询是最后一道保险。
        轮询失败时静默返回 False：宁可晚一步取消，也不能让 Redis 抖动阻断生成。
        """
        if self.cancel_event.is_set():
            return True
        store = _store()
        if store is None or not store.is_redis():
            return False
        now = time.monotonic()
        if now - self._last_poll < _POLL_INTERVAL_SECONDS:
            return False
        self._last_poll = now
        reason = store.cancel_flag(self.session_id)
        if reason:
            self.request_cancel(reason)
            return True
        return False

    def snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "status": self.status,
            "wait_ms": self.wait_ms,
            "cancel_reason": self.cancel_reason,
            "elapsed_ms": int((time.monotonic() - (self.started_at or self.queued_at)) * 1000),
        }


_LOCK = threading.Lock()
_ACTIVE: dict[str, TaskHandle] = {}


def _store() -> TaskStoreRouter:
    """取（并懒初始化）存储路由器；测试里可用 reset_store() 强制重建。"""
    global _STORE, _SUBSCRIBED
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                router = TaskStoreRouter()
                _STORE = router
                if router.is_redis() and not _SUBSCRIBED:
                    _SUBSCRIBED = True
                    _subscribe_cancel(router)
    return _STORE


def _apply_remote_cancel(session_id: str) -> None:
    """收到跨副本取消广播：把**本进程**对应的句柄置位。

    只在 redis 后端启用；广播是"提速"手段（不用等兜底轮询），
    丢失也不影响正确性（状态位 + should_stop 轮询兜底）。

    抽成模块级函数而不是内嵌闭包：这样能直接被测试调用，
    不必真的起一个订阅线程去制造广播（那种测法既慢又脆）。
    """
    with _LOCK:
        handle = _ACTIVE.get(session_id)
    if handle is not None and not handle.finished:
        handle.request_cancel("remote")
        metrics.incr("task.cancel.remote")
        logger.info("收到跨进程取消广播，已置位本地任务 | session=%s", session_id)


def _subscribe_cancel(router: TaskStoreRouter) -> None:
    router.subscribe_cancel(_apply_remote_cancel)


def reset_store() -> None:
    """重置存储路由器（测试用：换配置后需要重新按配置建后端）。"""
    global _STORE, _SUBSCRIBED
    with _STORE_LOCK:
        _STORE = None
        _SUBSCRIBED = False


def _fields(handle: TaskHandle) -> dict:
    """要共享出去的字段（与 snapshot() 同口径，不含对话内容）。

    时间戳用**墙钟**而不是 monotonic：monotonic 跨进程没有可比性，
    而共享状态要能被别的进程/运维看懂。
    """
    now = time.time()
    return {
        "session_id": handle.session_id,
        "run_id": handle.run_id,
        "status": handle.status,
        "wait_ms": handle.wait_ms,
        "cancel_reason": handle.cancel_reason,
        "queued_at": int(now - (handle.queued_at and (time.monotonic() - handle.queued_at) or 0)),
        "started_at": int(now) if handle.started_at is not None else None,
    }


def _mirror(handle: TaskHandle) -> None:
    """把本地句柄状态同步到存储（redis 模式下别的进程因此能看见它）。"""
    _store().save(handle.session_id, _fields(handle))


def register(session_id: str) -> TaskHandle:
    """登记一个新任务（进入排队态）。

    同一会话重复登记时替换旧句柄：一个会话同时只应有一轮在跑，
    旧句柄若是残留（例如进程异常），留着只会让 cancel 打到过期对象上。
    """
    handle = TaskHandle(session_id=session_id)
    with _LOCK:
        previous = _ACTIVE.get(session_id)
        if previous is not None and not previous.finished:
            logger.warning("会话已有在途任务，登记新任务并替换旧句柄 | %s", session_id)
            previous.request_cancel("superseded")
        _ACTIVE[session_id] = handle
    _mirror(handle)
    return handle


def current(session_id: str) -> TaskHandle | None:
    with _LOCK:
        return _ACTIVE.get(session_id)


def shared_status(session_id: str) -> str | None:
    """读**共享存储**里的状态（本地没有句柄时用）。

    用途：`/chat/cancel` 的响应里要报 status。若只读本地句柄，
    跨副本取消就会返回自相矛盾的 `{"cancelled": true, "status": "not_found"}`
    —— 取消其实已经写进共享状态并在对方进程生效了，调用方却被告知「没找到任务」。
    异常时返回 None（宁可少报状态，也不要让状态查询把取消接口搞挂）。
    """
    return _store().status_of(session_id)


def cancel(session_id: str, reason: str = "user") -> bool:
    """请求取消某会话的在途任务；没有在途任务返回 False。

    注意返回值语义：True 表示「取消信号已发出」，不等于「已经停下来」——
    线程要跑到下一个检查点才会真正结束，这一点在 API 文档里写清楚，
    免得调用方以为拿到 True 就可以立刻复用资源。

    跨进程：任务在**别的副本**上跑时，本地没有句柄也会往共享状态写取消位
    并广播 —— 这正是「cancel 打到另一个副本不再静默失败」的落点。
    """
    handle = current(session_id)
    if handle is None:
        store = _store()
        if store.is_redis() and store.status_of(session_id) not in (None, "done", "cancelled", "failed"):
            store.mark_cancelled(session_id, reason)
            metrics.incr("task.cancel.requested")
            logger.info("已请求取消**其他副本**的在途任务 | session=%s | reason=%s", session_id, reason)
            return True
        return False
    ok = handle.request_cancel(reason)
    if ok:
        _mirror(handle)
        _store().mark_cancelled(session_id, reason)
        metrics.incr("task.cancel.requested")
        logger.info("已请求取消任务 | session=%s | reason=%s", session_id, reason)
    return ok


def cancel_if_running(session_id: str, reason: str) -> bool:
    """仅当任务还在执行时请求取消（断连兜底用：正常跑完的任务不该被标成取消）。"""
    handle = current(session_id)
    if handle is not None:
        if handle.status != RUNNING:
            return False
        ok = handle.request_cancel(reason)
        if ok:
            _mirror(handle)
            _store().mark_cancelled(session_id, reason)
            metrics.incr(f"task.cancel.{reason}")
        return ok
    # 本地没有句柄：任务可能在另一个副本上，按共享状态判断（跨进程断连兜底）
    store = _store()
    if store.is_redis() and store.status_of(session_id) == RUNNING:
        store.mark_cancelled(session_id, reason)
        metrics.incr(f"task.cancel.{reason}")
        logger.info("已请求取消**其他副本**的在途任务 | session=%s | reason=%s", session_id, reason)
        return True
    return False


def backend() -> str:
    """当前任务状态后端（memory / redis），供 /health 展示降级是否发生。

    没有这个字段，"Redis 早就断了、现在跑的是进程内状态" 这件事就只能看启动日志，
    而日志早就滚过去了 —— 运维视角等于不可见。
    """
    return _store().backend


def counts() -> dict[str, int]:
    """排队 / 在执行的任务数（背压判断与 /health 都用它）。

    redis 模式下这是**全局**计数（含其他副本），背压因此按全局排队数判断，
    而不是"本副本没满就放行"——后者在多副本下实际并发是 N 倍。
    """
    return _store().counts()


def snapshot() -> dict:
    """在途任务明细（/health 用；只给状态与耗时，不含任何对话内容）。

    redis 模式下合并两路：共享存储里的在途（含**其他副本**的任务）
    + 本进程句柄的快照（信息更全：elapsed_ms 等本地派生字段）。
    """
    items: dict[str, dict] = {}
    for fields in _store().active():
        sid = fields.get("session_id")
        if sid:
            items[sid] = dict(fields)
    with _LOCK:
        for sid, handle in _ACTIVE.items():
            if not handle.finished:
                items[sid] = handle.snapshot()
    return {"counts": counts(), "active": list(items.values())}


def prune() -> int:
    """清理已结束的句柄，返回清理数量（长跑进程用，避免字典无限增长）。"""
    with _LOCK:
        stale = [sid for sid, h in _ACTIVE.items() if h.finished]
        for sid in stale:
            _ACTIVE.pop(sid, None)
    store = _store()
    for sid in stale:
        store.delete(sid)
    return len(stale)


def reset() -> None:
    """清空注册表（测试用）。"""
    with _LOCK:
        _ACTIVE.clear()
    reset_store()
