"""任务生命周期：排队、在途、取消。

此前一轮对话只有「开始」和「结束」两个状态，于是有两个真实问题：

1. **排队无界无反馈**：`await sem.acquire()` 会一直挂着，客户端只看到「卡住」，
   既不知道在排队、也不知道要等多久，更没法被拒绝；
2. **没有取消**：客户端断开（关标签页 / 断网）后，后台线程继续把这一轮跑完 ——
   用户看不到结果，token 却照烧（而这是用户付费的额度）。

这里把「一个正在跑的任务」显式建模：可查状态、可取消、可统计、可背压。

边界（必须说清）：状态在**进程内**，单副本部署足够；多副本要换成 Redis 之类的共享存储，
否则 cancel 请求打到另一个副本上会「找不到任务」而静默失败。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from . import metrics
from .logging_setup import get_logger

logger = get_logger(__name__)

#: 任务状态
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"
FAILED = "failed"

_TERMINAL = frozenset({DONE, CANCELLED, FAILED})


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

    # ------------------------------------------------------------ 状态
    def mark_running(self) -> int:
        """从排队进入执行，返回排队等待时长（毫秒）。"""
        self.started_at = time.monotonic()
        self.wait_ms = int((self.started_at - self.queued_at) * 1000)
        self.status = RUNNING
        return self.wait_ms

    def finish(self, status: str = DONE) -> None:
        """结束任务（已结束的任务再调用不会覆盖终态）。"""
        if self.status not in _TERMINAL:
            self.status = status

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
        """协作式停止信号：图执行层与流式推送都读它。"""
        return self.cancel_event.is_set()

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
    return handle


def current(session_id: str) -> TaskHandle | None:
    with _LOCK:
        return _ACTIVE.get(session_id)


def cancel(session_id: str, reason: str = "user") -> bool:
    """请求取消某会话的在途任务；没有在途任务返回 False。

    注意返回值语义：True 表示「取消信号已发出」，不等于「已经停下来」——
    线程要跑到下一个检查点才会真正结束，这一点在 API 文档里写清楚，
    免得调用方以为拿到 True 就可以立刻复用资源。
    """
    handle = current(session_id)
    if handle is None:
        return False
    ok = handle.request_cancel(reason)
    if ok:
        metrics.incr("task.cancel.requested")
        logger.info("已请求取消任务 | session=%s | reason=%s", session_id, reason)
    return ok


def cancel_if_running(session_id: str, reason: str) -> bool:
    """仅当任务还在执行时请求取消（断连兜底用：正常跑完的任务不该被标成取消）。"""
    handle = current(session_id)
    if handle is None or handle.status != RUNNING:
        return False
    ok = handle.request_cancel(reason)
    if ok:
        metrics.incr(f"task.cancel.{reason}")
    return ok


def counts() -> dict[str, int]:
    """排队 / 在执行的任务数（背压判断与 /health 都用它）。"""
    with _LOCK:
        waiting = sum(1 for h in _ACTIVE.values() if h.status == QUEUED)
        running = sum(1 for h in _ACTIVE.values() if h.status == RUNNING)
    return {"queued": waiting, "running": running, "total": waiting + running}


def snapshot() -> dict:
    """在途任务明细（/health 用；只给状态与耗时，不含任何对话内容）。"""
    with _LOCK:
        items = [h.snapshot() for h in _ACTIVE.values() if not h.finished]
    return {"counts": counts(), "active": items}


def prune() -> int:
    """清理已结束的句柄，返回清理数量（长跑进程用，避免字典无限增长）。"""
    with _LOCK:
        stale = [sid for sid, h in _ACTIVE.items() if h.finished]
        for sid in stale:
            _ACTIVE.pop(sid, None)
    return len(stale)


def reset() -> None:
    """清空注册表（测试用）。"""
    with _LOCK:
        _ACTIVE.clear()
