"""任务状态存储：进程内（默认）或 Redis（多副本共享）。

为什么单独一层：`tasks.py` 的公开 API（register / cancel / counts / snapshot）**一个都不能改**——
既有 473 条测试与「单副本部署行为逐字节兼容」的老约定都压在它上面。所以把
「状态存哪、取消怎么传到别的进程」整层抽到这里，`tasks.py` 只做转发。

三个设计取舍（都是被真实故障逼出来的，不是偏好）：

1. **全局计数由状态派生，不维护独立计数器**。直觉写法是「排队 +1、结束 -1」，
   但异常路径一旦漏减，计数会虚增，最终把服务**永久**堵死（谁也进不来）。
   派生计数（扫在途键按 status 汇总）天然不会漂移，代价是 O(在途任务数)——
   演示与单机规模完全够用；若将来任务量上千，再换成 Lua 维护计数器并接受漏减风险。

2. **取消必须三保险，不能只靠 Pub/Sub**。Redis 发布订阅**不保证送达**：
   订阅者不在线时消息直接丢。所以取消 = ① 写共享 `cancel_reason` 状态位
   ② Pub/Sub 广播（让**在线**副本立即响应，不用等轮询）③ `should_stop()`
   兜底轮询（消息丢了 / 订阅没起来时仍能取消）。只靠 ② 会漏，只靠 ③ 会慢。

3. **共享状态不落对话内容**。字段与 `tasks.snapshot()` 现有口径一致：
   session_id / run_id / status / 时间戳 / 取消原因。这是隐私红线。

路由（按配置与连通性在两者间选择、故障降级、恢复切回）见 `TaskStoreRouter`。
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod

from . import config
from .logging_setup import get_logger

logger = get_logger(__name__)

#: 任务状态字段（与 tasks.py 的 TaskHandle.snapshot() 同口径，缺一不可）
FIELDS = ("session_id", "run_id", "status", "queued_at", "started_at", "wait_ms", "cancel_reason")


class TaskStore(ABC):
    """任务状态存储接口：字段级读写 + 取消传播 + 全局统计。"""

    #: 后端名（memory / redis），供日志与 /health 展示
    backend: str = "?"

    # ------------------------------------------------------------ 状态
    @abstractmethod
    def save(self, session_id: str, fields: dict) -> None:
        """写入（或更新）一个任务的状态字段。"""

    @abstractmethod
    def status_of(self, session_id: str) -> str | None:
        """读取状态；不存在返回 None。"""

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """删除任务状态（终态清理用）。"""

    @abstractmethod
    def active(self) -> list[dict]:
        """全部在途任务的字段快照（/health 用；只含状态与耗时）。"""

    @abstractmethod
    def counts(self) -> dict[str, int]:
        """全局 queued / running / total（由状态派生，不维护计数器）。"""

    # ------------------------------------------------------------ 取消
    @abstractmethod
    def mark_cancelled(self, session_id: str, reason: str) -> None:
        """写共享取消状态位（持久、跨进程可见）。"""

    @abstractmethod
    def cancel_flag(self, session_id: str) -> str | None:
        """读共享取消位；未取消返回 None（兜底轮询用）。"""

    @abstractmethod
    def publish_cancel(self, session_id: str) -> None:
        """广播取消通知（尽力而为：不保证送达，所以上面还有状态位兜底）。"""

    # ------------------------------------------------------------ 连通性
    @abstractmethod
    def ping(self) -> bool:
        """探活：能读能写才算可用（只 PING 会被只读副本骗过）。"""

    def subscribe_cancel(self, callback) -> None:
        """订阅取消广播，收到就把 session_id 交给 callback（尽力而为，可多次调用）。

        memory 后端无需订阅（没有"另一个进程"要通知）。
        Redis 后端起一个守护线程：**不保证送达**，所以调用方仍须有
        `cancel_flag()` 兜底轮询（见模块 docstring 的"三保险"）。
        """
        return


class MemoryTaskStore(TaskStore):
    """进程内存储：默认后端，单副本部署行为与历史版本完全一致。

    这里刻意**不引入任何新机制**（就是一个带锁的字典），目的是让 memory 模式
    成为"零风险默认"：没装 Redis 的机器走这条路，行为与改造前逐字节相同。
    """

    backend = "memory"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, dict] = {}

    def save(self, session_id: str, fields: dict) -> None:
        with self._lock:
            self._state[session_id] = dict(fields)

    def status_of(self, session_id: str) -> str | None:
        with self._lock:
            item = self._state.get(session_id)
            return None if item is None else item.get("status")

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._state.pop(session_id, None)

    def active(self) -> list[dict]:
        with self._lock:
            return [dict(v) for v in self._state.values() if v.get("status") not in _TERMINAL]

    def counts(self) -> dict[str, int]:
        with self._lock:
            waiting = sum(1 for v in self._state.values() if v.get("status") == "queued")
            running = sum(1 for v in self._state.values() if v.get("status") == "running")
        return {"queued": waiting, "running": running, "total": waiting + running}

    def mark_cancelled(self, session_id: str, reason: str) -> None:
        with self._lock:
            item = self._state.get(session_id)
            if item is not None:
                item["cancel_reason"] = reason

    def cancel_flag(self, session_id: str) -> str | None:
        with self._lock:
            item = self._state.get(session_id)
            return None if item is None else item.get("cancel_reason") or None

    def publish_cancel(self, session_id: str) -> None:
        """进程内没有"另一个副本"要通知，状态位已经足够。"""

    def subscribe_cancel(self, callback) -> None:
        """进程内无需订阅：取消就发生在本进程，置位本地 Event 即可。"""
        return

    def ping(self) -> bool:
        return True


#: 终态集合（与 tasks.py 的 _TERMINAL 保持一致；这里独立定义避免循环导入）
_TERMINAL = frozenset({"done", "cancelled", "failed"})


class RedisTaskStore(TaskStore):
    """Redis 存储：跨副本共享状态、广播取消、派生全局计数。

    数据结构（全部带 TTL，终态靠 TTL 自然清理，本轮不做历史持久化）：
    - `{prefix}{session_id}` → Hash：run_id / status / queued_at / started_at /
      wait_ms / cancel_reason（取消状态位）
    - `{prefix}cancel` → Pub/Sub 频道：广播 "session_id"

    为什么不用 Redis Stream / List 做队列：任务本身是**内联执行**的
    （同步 /chat 与 SSE /chat/stream 在请求里跑完），没有"待处理任务"需要暂存；
    这里要共享的只是**状态与信号**，Hash + Pub/Sub 足够且语义更简单。
    """

    backend = "redis"

    def __init__(self, url: str, prefix: str, ttl_seconds: int, client=None) -> None:
        if client is not None:
            # 显式注入：单测用假客户端即可覆盖全部读写路径，本机不必装 Redis
            self._redis = client
        else:
            import redis  # 延迟导入：没装可选依赖时 memory 模式照常可用

            self._redis = redis.Redis.from_url(
                        url,
                        decode_responses=True,
                        socket_timeout=2.0,
                        socket_connect_timeout=2.0,
                        health_check_interval=15,
                        # 必须显式锁 RESP2（2026-10-16 实测踩到）：redis-py 5.0+ 默认在连接时发
                        # HELLO 握手切 RESP3，而 HELLO 是 Redis 6.0 才有的命令。碰到 5.x 的
                        # 服务端（本机 Windows 版 Redis 5.0.14 就是）会直接
                        # `ResponseError: unknown command HELLO` 连不上——而且现象是"客户端装新版就崩"，
                        # 很容易误判成服务端挂了。本项目不需要 RESP3 的任何特性，锁 RESP2 最稳。
                        protocol=2,
                    )
        self._prefix = prefix
        self._ttl = max(60, int(ttl_seconds))

    # ------------------------------------------------------------ 状态
    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def save(self, session_id: str, fields: dict) -> None:
        payload = {k: v for k, v in fields.items() if v is not None}
        pipe = self._redis.pipeline()
        pipe.hset(self._key(session_id), mapping={k: str(v) for k, v in payload.items()})
        pipe.expire(self._key(session_id), self._ttl)
        pipe.execute()

    def status_of(self, session_id: str) -> str | None:
        return self._redis.hget(self._key(session_id), "status")

    def delete(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))

    def active(self) -> list[dict]:
        items: list[dict] = []
        for key in self._redis.scan_iter(match=f"{self._prefix}*", count=100):
            if key.endswith(":cancel") or key.endswith(":cancel-ping"):
                continue
            data = self._redis.hgetall(key)
            if not data:
                continue
            status = data.get("status", "")
            if status in _TERMINAL:
                continue
            items.append(_normalize(data))
        return items

    def counts(self) -> dict[str, int]:
        waiting = running = 0
        for item in self.active():
            if item.get("status") == "queued":
                waiting += 1
            elif item.get("status") == "running":
                running += 1
        return {"queued": waiting, "running": running, "total": waiting + running}

    # ------------------------------------------------------------ 取消
    def mark_cancelled(self, session_id: str, reason: str) -> None:
        """写共享取消状态位并广播 —— 两者是同一个动作，不拆开调用。

        拆开的话，调用方一旦只写不广播（或者只广播不写），就会出现
        "广播出去了但没有状态位"（别的进程收到消息却查不到任务）或反之，
        而这类不一致只在多副本下才暴露，排查成本极高。
        广播失败只影响响应速度，状态位已落库，should_stop 兜底轮询仍能取消。
        """
        key = self._key(session_id)
        pipe = self._redis.pipeline()
        pipe.hset(key, "cancel_reason", reason)
        pipe.expire(key, self._ttl)
        pipe.execute()
        self.publish_cancel(session_id)

    def cancel_flag(self, session_id: str) -> str | None:
        # 归一化空串 → None（与 MemoryTaskStore 同语义）：save() 会把 cancel_reason=""
        # 一起写进 Hash，直接返回 hget 的结果就会出现"memory 返回 None、redis 返回 ''"
        # 的不一致 —— 虽然空串在 if 判断里同样为假，但两个实现行为不同，
        # 就会写出只在其中一个后端上通过的测试，切换后端即失效。
        return self._redis.hget(self._key(session_id), "cancel_reason") or None

    def publish_cancel(self, session_id: str) -> None:
        # 失败只影响"响应速度"，不影响正确性（状态位兜底）
        self._redis.publish(f"{self._prefix}cancel", session_id)

    def subscribe_cancel(self, callback) -> None:
        """起守护线程订阅取消广播；连接失败只记日志（兜底轮询仍有效）。"""
        import threading

        def _loop() -> None:
            # pubsub 必须先占位：若 self._redis.pubsub() 自己就抛了，
            # finally 里的 pubsub.close() 会踩 UnboundLocalError，把真正的失败原因盖掉。
            pubsub = None
            try:
                pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(f"{self._prefix}cancel")
                for message in pubsub.listen():
                    session_id = message.get("data") if isinstance(message, dict) else None
                    if isinstance(session_id, str) and session_id:
                        try:
                            callback(session_id)
                        except Exception as exc:  # noqa: BLE001 - 回调异常绝不能杀死订阅线程
                            logger.debug("取消广播回调异常已忽略：%s", exc)
            except Exception as exc:  # noqa: BLE001 - 连接断了就退出，等下次重启/恢复
                logger.warning("取消广播订阅中断（将依赖 should_stop 兜底轮询）：%s", exc)
            finally:
                if pubsub is not None:
                    try:
                        pubsub.close()
                    except Exception:  # noqa: BLE001 - 关闭失败无所谓，连接已断
                        pass

        thread = threading.Thread(target=_loop, name="task-cancel-sub", daemon=True)
        thread.start()
        return

    # ------------------------------------------------------------ 连通性
    def ping(self) -> bool:
        probe_key = f"{self._prefix}cancel-ping"
        self._redis.set(probe_key, "1", ex=10)
        back = self._redis.get(probe_key)
        self._redis.delete(probe_key)
        return back == "1"


def _normalize(data: dict) -> dict:
    """把 Redis Hash 的字符串值转回可 JSON 序列化的形态（/health 要输出）。"""
    out: dict = {}
    for key in FIELDS:
        if key not in data:
            continue
        raw = data[key]
        if key in ("queued_at", "started_at", "wait_ms") and raw not in (None, ""):
            try:
                out[key] = int(float(raw))
            except (TypeError, ValueError):
                out[key] = 0
        else:
            out[key] = raw
    return out


def build_store() -> TaskStore:
    """按配置构造存储实例（**不探活**）。

    为什么单独拆出来：早期版本在这里顺带 ping，探活失败就让构造抛异常，
    于是路由里拿不到实例、恢复探测也就永远无从下手 ——
    "自动恢复"变成了只有"启动成功、中途才坏"才成立的半截功能。
    构造与探活必须分开：实例留着，才能在 Redis 恢复后重新探活并切回。
    """
    backend = config.TASK_STATE_BACKEND
    if backend == "redis":
        return RedisTaskStore(
            config.TASK_STATE_REDIS_URL, config.TASK_STATE_KEY_PREFIX, config.TASK_STATE_TTL_SECONDS
        )
    if backend != "memory":
        logger.warning("未知 TASK_STATE_BACKEND=%s，回退为进程内 memory", backend)
    return MemoryTaskStore()


class TaskStoreRouter:
    """按配置与连通性选择存储后端，故障降级、按间隔探测恢复。

    与 jobseeker 三件套同一套取舍（那边是 Java，这边是 Python）：
    - 默认 memory：不装 Redis 的机器**不建连接**，启动零噪音；
    - 启用但连不上 → 启动即降级 + 告警，绝不静默；
    - 运行中故障 → 当次操作转交 memory 并告警（同一故障只告警一次）；
    - 恢复 → 探测成功后切回并明确告知。
    """

    def __init__(self) -> None:
        self._memory = MemoryTaskStore()
        self._redis: TaskStore | None = None
        self._warned = False
        self._last_probe = 0.0
        self._active: TaskStore = self._memory

        if config.TASK_STATE_BACKEND != "redis":
            logger.info("任务状态后端 = 进程内 memory（TASK_STATE_BACKEND=%s）", config.TASK_STATE_BACKEND)
            return

        # 构造与探活**分开处理**：探活失败必须保留实例，否则 Redis 恢复后无从探活，
        # 「自动恢复」就成了只有"启动成功、中途才坏"才成立的半截功能。
        try:
            store = build_store()
        except Exception as exc:  # noqa: BLE001 - 启动期任何失败都降级，不阻断服务
            self._active = self._memory
            logger.warning(
                "任务状态后端 = 进程内 memory（Redis 已启用但不可用：%s；%ds 后自动重试）",
                exc,
                config.TASK_STATE_HEALTH_CHECK_SECONDS,
            )
            return

        self._redis = store
        self._active = self._memory
        try:
            healthy = store.ping()
        except Exception as exc:  # noqa: BLE001 - 探活异常同样只降级，实例必须留着
            healthy = False
            logger.debug("启动期 Redis 探活异常：%s", exc)
        if healthy:
            self._active = store
            logger.info("任务状态后端 = Redis（跨副本共享：%s）", config.TASK_STATE_REDIS_URL)
        else:
            logger.warning(
                "任务状态后端 = 进程内 memory（Redis 已启用但探活失败：%s；%ds 后自动重试）",
                config.TASK_STATE_REDIS_URL,
                config.TASK_STATE_HEALTH_CHECK_SECONDS,
            )

    # ------------------------------------------------------------ 查询
    @property
    def backend(self) -> str:
        return self._active.backend

    def is_redis(self) -> bool:
        return self._active.backend == "redis"

    def _try_recover(self) -> None:
        """按间隔尝试恢复；未启用 / 已健康时零开销。"""
        if self._redis is None or self._active is self._redis:
            return
        now = time.monotonic()
        if now - self._last_probe < max(1, config.TASK_STATE_HEALTH_CHECK_SECONDS):
            return
        self._last_probe = now
        try:
            if self._redis.ping():
                self._active = self._redis
                self._warned = False
                logger.info("任务状态已切回 Redis 后端")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis 恢复探测仍失败：%s", exc)

    def _degrade(self, what: str) -> None:
        # 判据是「当前用的不是 memory 就换到 memory」，**不是**「当前用的恰好是 _redis 实例」：
        # 后者在 active 指向别的实例时（测试注入、将来重构、降级与恢复的中间态）
        # 会静默不降级 —— 实测过一次，表现为 save/status 继续打向已故障的后端、
        # 读操作直接失败，而日志里只有一条"已降级"的告警，极难察觉。
        if self._active is not self._memory:
            self._active = self._memory
        if not self._warned:
            self._warned = True
            logger.warning("任务状态 %s 失败，已降级为进程内 memory（稍后自动尝试恢复）", what)

    # ------------------------------------------------------------ 操作（带降级）
    def save(self, session_id: str, fields: dict) -> None:
        self._try_recover()
        try:
            self._active.save(session_id, fields)
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"写入（{type(exc).__name__}）")
            self._memory.save(session_id, fields)

    def status_of(self, session_id: str) -> str | None:
        self._try_recover()
        try:
            return self._active.status_of(session_id)
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"读取（{type(exc).__name__}）")
            return self._memory.status_of(session_id)

    def delete(self, session_id: str) -> None:
        self._try_recover()
        try:
            self._active.delete(session_id)
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"删除（{type(exc).__name__}）")

    def counts(self) -> dict[str, int]:
        self._try_recover()
        try:
            return self._active.counts()
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"统计（{type(exc).__name__}）")
            return self._memory.counts()

    def active(self) -> list[dict]:
        self._try_recover()
        try:
            return self._active.active()
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"快照（{type(exc).__name__}）")
            return self._memory.active()

    # ------------------------------------------------------------ 取消（三保险）
    def mark_cancelled(self, session_id: str, reason: str) -> None:
        self._try_recover()
        try:
            self._active.mark_cancelled(session_id, reason)
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"取消（{type(exc).__name__}）")
            # 降级后仍要把取消写进 memory：否则本进程的 should_stop 兜底也失效
            self._memory.mark_cancelled(session_id, reason)

    def cancel_flag(self, session_id: str) -> str | None:
        """兜底轮询：读共享取消位。异常时返回 None（宁可漏一次，也不能阻断生成）。"""
        try:
            return self._active.cancel_flag(session_id)
        except Exception:  # noqa: BLE001 - 轮询失败静默：下次还会试
            return None

    # ------------------------------------------------------------ 订阅
    def subscribe_cancel(self, callback) -> None:
        """订阅取消广播（转发给当前后端）。

        这个方法**必须**存在于路由器上：订阅是在 `tasks._store()` 里首次用到存储时发起的，
        而那里拿到的是路由器而不是具体后端。漏掉它时症状特别隐蔽——
        单测里用的替身自带同名方法，测试全绿；真机上第一次 `/health` 就 500
        （AttributeError: 'TaskStoreRouter' object has no attribute 'subscribe_cancel'）。
        """
        self._try_recover()
        try:
            self._active.subscribe_cancel(callback)
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"订阅（{type(exc).__name__}）")
            self._memory.subscribe_cancel(callback)
