"""任务状态存储的测试：进程内默认路径 + Redis 跨副本路径 + 故障降级。

本机没装 Redis，所以 Redis 路径用一个**假客户端**（FakeRedis，纯内存）覆盖：
它保证「存进去能被另一个 store 读出来」这条跨副本语义能被真正验到 ——
用 mock 断言调用次数只能证明"调过"，证明不了"另一个进程真的看得见"。

三条必须守住的不变式：
1. 默认（memory）行为与改造前一致；
2. 取消**跨副本**可见（共享状态位）；
3. Redis 出故障时自动回退 memory，请求照常完成（可用性优先于严格性）。
"""

from __future__ import annotations

import fnmatch
import threading
import time

import pytest

from src import config
from src.tasks_store import (
    MemoryTaskStore,
    RedisTaskStore,
    TaskStoreRouter,
    build_store,
)


# --------------------------------------------------------------------- 假客户端
class FakeRedis:
    """最小可用的 Redis 替身：只实现 tasks_store 真正用到的那几个命令。"""

    def __init__(self, *, broken: bool = False) -> None:
        self.hashes: dict[str, dict] = {}
        self.ttl: dict[str, int] = {}
        self.plain: dict[str, str] = {}
        self.published: list[tuple[str, str]] = []
        self.broken = broken  # True 时所有命令抛异常，模拟 Redis 挂了
        self.pubsub_messages: list[dict] = []
        self.pubsub_error: Exception | None = None

    def _check(self) -> None:
        if self.broken:
            raise ConnectionError("connection refused")

    # ---- 单条命令
    def hset(self, key, field=None, value=None, mapping=None):
        self._check()
        bucket = self.hashes.setdefault(key, {})
        if field is not None:
            bucket[field] = str(value)
        for k, v in (mapping or {}).items():
            bucket[k] = str(v)
        return 1

    def hget(self, key, field):
        self._check()
        return self.hashes.get(key, {}).get(field)

    def hgetall(self, key):
        self._check()
        return dict(self.hashes.get(key, {}))

    def delete(self, key):
        self._check()
        self.hashes.pop(key, None)
        self.plain.pop(key, None)
        self.ttl.pop(key, None)
        return 1

    def expire(self, key, ttl):
        self._check()
        self.ttl[key] = int(ttl)
        return True

    def scan_iter(self, match="*", count=None):
        self._check()
        for key in list(self.hashes):
            if fnmatch.fnmatch(key, match):
                yield key

    def publish(self, channel, message):
        self._check()
        self.published.append((channel, str(message)))
        return 1

    def set(self, key, value, ex=None):
        self._check()
        self.plain[key] = str(value)
        if ex:
            self.ttl[key] = int(ex)
        return True

    def get(self, key):
        self._check()
        return self.plain.get(key)

    def pipeline(self):
        return FakePipeline(self)

    def pubsub(self, ignore_subscribe_messages=True):
        self._check()
        if self.pubsub_error is not None:
            raise self.pubsub_error
        return FakePubSub(self.pubsub_messages)

    def ping(self):
        self._check()
        return True


class FakePubSub:
    """订阅端替身：按预设消息列表依次吐出，然后结束（等价于连接断开）。"""

    def __init__(self, messages: list[dict]) -> None:
        self._messages = list(messages)
        self.channel: str | None = None
        self.closed = False

    def subscribe(self, channel):
        self.channel = channel

    def listen(self):
        yield from self._messages

    def close(self):
        self.closed = True


class FakePipeline:
    """pipeline 只是把命令攒起来再一起执行 —— 这里按顺序回放即可。"""

    def __init__(self, client: FakeRedis) -> None:
        self._client = client
        self._queued: list[tuple[str, tuple, dict]] = []

    def hset(self, *args, **kwargs):
        self._queued.append(("hset", args, kwargs))
        return self

    def expire(self, *args, **kwargs):
        self._queued.append(("expire", args, kwargs))
        return self

    def execute(self):
        for name, args, kwargs in self._queued:
            getattr(self._client, name)(*args, **kwargs)
        self._queued.clear()
        return True


@pytest.fixture(autouse=True)
def _pin_backend_to_memory(monkeypatch):
    """把 TASK_STATE_BACKEND 钉成 memory，避免测试被**本机 .env** 影响。

    背景（2026-10-16）：给 `.env` 打开 `TASK_STATE_BACKEND=redis` 之后，
    4 条"默认后端是进程内"的用例在本地全红、CI 却全绿 —— 因为 `.env` 被 gitignore，
    CI 上根本没有这个文件。根因是这些用例**隐式依赖默认值**，而不是显式声明前提。
    凡是"默认应为 X"的测试，都必须自己把 X 写出来，否则环境一变就失效。
    """
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "memory")


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def redis_store(fake_redis):
    return RedisTaskStore("redis://unused/0", "agent:task:", 600, client=fake_redis)


# --------------------------------------------------------------------- memory
def test_memory_store_round_trip():
    store = MemoryTaskStore()
    store.save("s1", {"session_id": "s1", "status": "queued"})
    assert store.status_of("s1") == "queued"
    assert store.cancel_flag("s1") is None
    store.mark_cancelled("s1", "user")
    assert store.cancel_flag("s1") == "user"
    store.delete("s1")
    assert store.status_of("s1") is None
    assert store.ping() is True
    assert store.backend == "memory"


def test_memory_counts_derived_from_state():
    """计数必须**由状态派生**：终态不计入，否则跑久了计数只增不减。"""
    store = MemoryTaskStore()
    store.save("s1", {"session_id": "s1", "status": "queued"})
    store.save("s2", {"session_id": "s2", "status": "running"})
    store.save("s3", {"session_id": "s3", "status": "queued"})
    store.save("s4", {"session_id": "s4", "status": "done"})
    store.save("s5", {"session_id": "s5", "status": "cancelled"})

    counts = store.counts()
    assert counts == {"queued": 2, "running": 1, "total": 3}
    assert {i["session_id"] for i in store.active()} == {"s1", "s2", "s3"}


# --------------------------------------------------------------------- redis
def test_redis_store_round_trip(redis_store, fake_redis):
    redis_store.save("s1", {"session_id": "s1", "run_id": "r1", "status": "queued", "wait_ms": 0})
    assert redis_store.status_of("s1") == "queued"
    assert "agent:task:s1" in fake_redis.hashes, "key 必须带前缀，避免与其他业务串味"
    assert fake_redis.ttl["agent:task:s1"] == 600, "必须带 TTL，否则终态任务永远留在 Redis 里"


def test_cancel_is_visible_across_replicas(fake_redis):
    """**核心用例**：A 副本发起取消，B 副本必须看得到。

    这正是改造前的静默失败：cancel 打到另一个副本时找不到本地句柄就返回 False，
    用户以为停了，token 照烧。
    """
    replica_a = RedisTaskStore("redis://x/0", "agent:task:", 600, client=fake_redis)
    replica_b = RedisTaskStore("redis://x/0", "agent:task:", 600, client=fake_redis)

    replica_a.save("sess-7", {"session_id": "sess-7", "status": "running"})
    # B 副本本地没有这个会话（它在 A 上跑），但共享状态里看得见
    assert replica_b.status_of("sess-7") == "running"
    assert replica_b.cancel_flag("sess-7") is None

    replica_a.mark_cancelled("sess-7", "user")
    assert replica_b.cancel_flag("sess-7") == "user", "取消必须跨副本可见"
    assert ("agent:task:cancel", "sess-7") in fake_redis.published, "还要广播，让在线副本立即响应"


def test_counts_are_global_across_replicas(fake_redis):
    """两个副本各排一个 → 背压看到的是全局 2，而不是各自 1。"""
    a = RedisTaskStore("redis://x/0", "agent:task:", 600, client=fake_redis)
    b = RedisTaskStore("redis://x/0", "agent:task:", 600, client=fake_redis)
    a.save("s1", {"session_id": "s1", "status": "queued"})
    b.save("s2", {"session_id": "s2", "status": "queued"})

    assert a.counts()["queued"] == 2
    assert b.counts()["queued"] == 2
    b.save("s2", {"session_id": "s2", "status": "done"})
    assert a.counts()["queued"] == 1, "B 结束的任务，A 的背压计数要跟着降"


def test_redis_active_normalizes_numbers(redis_store):
    redis_store.save("s1", {"session_id": "s1", "status": "running", "queued_at": 1000, "started_at": 1010})
    item = redis_store.active()[0]
    assert item["queued_at"] == 1000 and isinstance(item["queued_at"], int)
    assert item["started_at"] == 1010


# --------------------------------------------------------------------- 路由与降级
def test_router_defaults_to_memory(monkeypatch):
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "memory")
    router = TaskStoreRouter()
    assert router.backend == "memory"
    assert router.is_redis() is False


def test_router_unknown_backend_falls_back(monkeypatch):
    """拼错配置名不能让它崩掉，也不能当成 redis 去连。"""
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "rediis")
    router = TaskStoreRouter()
    assert router.backend == "memory"


def _patched_build_store(monkeypatch, client):
    """替换 build_store，返回一个接在假客户端上的 Redis 存储。

    注意探活这一步**故意不在这里做** —— 路由自己会 ping，
    这样测的才是真实路径（早期版本把 ping 藏在构造里，测试就绕过了它 ✗）。
    """
    monkeypatch.setattr(
        "src.tasks_store.build_store",
        lambda: RedisTaskStore(config.TASK_STATE_REDIS_URL, "agent:task:", 600, client=client),
    )


def test_router_degrades_when_redis_broken(monkeypatch):
    """Redis 挂着 → 自动回退 memory，请求照常能跑（可用性优先）。"""
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "redis")
    monkeypatch.setattr(config, "TASK_STATE_HEALTH_CHECK_SECONDS", 9999)
    _patched_build_store(monkeypatch, FakeRedis(broken=True))

    router = TaskStoreRouter()
    assert router.backend == "memory", "探活失败必须在启动期就降级，而不是等第一次请求"

    router.save("s1", {"session_id": "s1", "status": "queued"})
    assert router.status_of("s1") == "queued", "降级后写入走 memory，功能不丢"
    assert router.counts()["queued"] == 1


def test_router_recovers_when_redis_comes_back(monkeypatch):
    """Redis 恢复后要自动切回，否则一次抖动就永久降级。"""
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "redis")
    monkeypatch.setattr(config, "TASK_STATE_HEALTH_CHECK_SECONDS", 0)
    client = FakeRedis(broken=True)
    _patched_build_store(monkeypatch, client)

    router = TaskStoreRouter()
    assert router.backend == "memory", "探活失败应先降级"

    client.broken = False  # Redis 恢复
    router._try_recover()
    assert router.backend == "redis", "恢复后必须自动切回 Redis"


def test_build_store_honours_backend(monkeypatch):
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "memory")
    assert isinstance(build_store(), MemoryTaskStore)


def test_cancel_flag_swallows_errors(monkeypatch):
    """兜底轮询失败必须静默返回 None：宁可晚一步取消，也不能让 Redis 抖动阻断生成。"""
    router = TaskStoreRouter()
    router._active = router._memory
    broken = RedisTaskStore("redis://x/0", "agent:task:", 600, client=FakeRedis(broken=True))
    router._active = broken
    assert router.cancel_flag("s1") is None


# --------------------------------------------------------------------- 与 tasks.py 的衔接
def test_should_stop_polls_shared_flag_when_broadcast_missed():
    """三保险的第三道：广播没送达时，should_stop() 靠共享状态位仍能取消。

    模拟真实故障：另一个副本写了取消状态位，但本进程的 Pub/Sub 订阅没收到
    （订阅者不在线 / 连接中断）。若只靠广播，这次生成会一直跑到底、token 照烧。
    """
    from src import tasks

    client = FakeRedis()
    router = TaskStoreRouter()
    router._active = RedisTaskStore("redis://x/0", "agent:task:", 600, client=client)
    tasks.reset()
    tasks._STORE = router

    handle = tasks.register("sess-1")
    handle.mark_running()
    assert handle.should_stop() is False, "没取消时不得误报停止"

    # 另一个副本只写状态位，没有广播
    client.hset("agent:task:sess-1", "cancel_reason", "user")
    handle._last_poll = 0.0  # 绕过轮询节流，确保这次真的读一次
    assert handle.should_stop() is True, "共享状态位必须被兜底读到"
    assert handle.cancel_reason == "user", "读到后要置位本地 Event，后续检查点不用再查 Redis"
    assert handle.should_stop() is True, "本地 Event 已置位，不必再依赖 Redis"
    tasks.reset()


def test_memory_mode_never_touches_redis():
    """默认 memory：should_stop 只看本地 Event，绝不去连 Redis（零风险默认）。"""
    from src import tasks

    tasks.reset()
    tasks._STORE = TaskStoreRouter()  # memory 后端
    handle = tasks.register("sess-2")
    handle.mark_running()
    assert handle.should_stop() is False
    handle.request_cancel("user")
    assert handle.should_stop() is True
    tasks.reset()


# --------------------------------------------------------------------- 订阅广播
def test_subscribe_cancel_invokes_callback(fake_redis):
    """广播到达时要真的回调：在线副本据此立即置位本地 Event（不用等兜底轮询）。"""
    fake_redis.pubsub_messages = [{"type": "message", "data": "sess-9"}]
    store = RedisTaskStore("redis://x/0", "agent:task:", 600, client=fake_redis)

    got = []
    done = threading.Event()

    def _cb(session_id):
        got.append(session_id)
        done.set()

    store.subscribe_cancel(_cb)
    assert done.wait(timeout=5), "订阅线程应在超时前收到广播并回调"
    assert got == ["sess-9"]


def test_subscribe_cancel_survives_callback_error(fake_redis):
    """回调抛异常绝不能杀死订阅线程（否则一次脏数据就让取消永久失效）。"""
    fake_redis.pubsub_messages = [{"type": "message", "data": "sess-a"}]
    store = RedisTaskStore("redis://x/0", "agent:task:", 600, client=fake_redis)

    done = threading.Event()

    def _boom(session_id):
        done.set()
        raise ValueError("回调炸了")

    store.subscribe_cancel(_boom)
    assert done.wait(timeout=5)


def test_subscribe_cancel_survives_connection_error(fake_redis):
    """订阅连接断开时不抛到调用方：应退化成"只靠兜底轮询"，而不是启动失败。"""
    fake_redis.pubsub_error = ConnectionError("connection refused")
    store = RedisTaskStore("redis://x/0", "agent:task:", 600, client=fake_redis)

    store.subscribe_cancel(lambda sid: None)  # 不抛异常即可
    time.sleep(0.2)  # 给线程一点时间走到 except 分支


def test_memory_subscribe_is_noop():
    """memory 后端订阅是空操作：进程内没有"另一个副本"要通知。"""
    assert MemoryTaskStore().subscribe_cancel(lambda sid: None) is None


# --------------------------------------------------------------------- 故障路径全覆盖
class _ExplodingStore:
    """每个方法都抛异常，用来覆盖路由器各操作的降级分支。"""

    backend = "redis"

    def _boom(self, *_args, **_kwargs):
        raise ConnectionError("redis down")

    save = status_of = delete = counts = active = mark_cancelled = ping = _boom


def test_router_degrades_every_operation_on_error(monkeypatch):
    """Redis 中途挂掉：写/读/统计/快照/删除/取消全部要降级，且请求照常完成。"""
    monkeypatch.setattr(config, "TASK_STATE_HEALTH_CHECK_SECONDS", 9999)
    router = TaskStoreRouter()
    exploding = _ExplodingStore()
    router._active = exploding
    router._redis = exploding

    router.save("s1", {"session_id": "s1", "status": "queued"})
    assert router.status_of("s1") == "queued"
    assert router.counts()["queued"] == 1
    assert router.active()
    router.mark_cancelled("s1", "user")
    assert router.cancel_flag("s1") == "user", "降级后取消仍要落到 memory，否则本进程兜底也失效"
    router.delete("s1")
    assert router.status_of("s1") is None
    assert router.backend == "memory", "故障后必须降级为 memory"


def test_router_warns_only_once_per_outage(monkeypatch, caplog):
    """同一故障只告警一次：否则每次请求刷一条 WARN，高峰期把日志刷满、真异常被埋掉。"""
    monkeypatch.setattr(config, "TASK_STATE_HEALTH_CHECK_SECONDS", 9999)
    router = TaskStoreRouter()
    router._active = _ExplodingStore()

    with caplog.at_level("WARNING", logger="src.tasks_store"):
        for i in range(5):
            router.save(f"s{i}", {"session_id": f"s{i}", "status": "queued"})
    warnings = [rec for rec in caplog.records if "降级" in rec.getMessage()]
    assert len(warnings) == 1, f"5 次失败只应告警一次，实际 {len(warnings)} 次"


def test_router_constructor_survives_build_failure(monkeypatch):
    """构造存储本身失败（缺可选依赖 / URL 非法）也必须降级，不能让服务起不来。"""
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "redis")

    def _explode():
        raise ModuleNotFoundError("No module named 'redis'")

    monkeypatch.setattr("src.tasks_store.build_store", _explode)
    router = TaskStoreRouter()
    assert router.backend == "memory"
    assert router.is_redis() is False


def test_router_survives_ping_exception_at_startup(monkeypatch):
    """探活**抛异常**（不是返回 False）也只降级，且实例必须留着以便日后恢复。"""
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "redis")
    monkeypatch.setattr(config, "TASK_STATE_HEALTH_CHECK_SECONDS", 9999)
    store = _ExplodingStore()
    monkeypatch.setattr("src.tasks_store.build_store", lambda: store)

    router = TaskStoreRouter()
    assert router.backend == "memory"
    assert router._redis is store, "探活异常不能把实例丢掉，否则永远无法自动恢复"


def test_normalize_tolerates_bad_numbers(redis_store):
    """共享状态里的脏数据不能把 /health 整个打挂：解析不了就按 0 处理。"""
    redis_store._redis.hashes["agent:task:s1"] = {
        "session_id": "s1",
        "status": "running",
        "queued_at": "不是数字",
    }
    item = redis_store.active()[0]
    assert item["queued_at"] == 0


def test_redis_active_skips_pubsub_and_ping_keys(redis_store, fake_redis):
    """广播频道名/探针键不是任务，统计时必须跳过，否则 counts 会算错。"""
    fake_redis.hashes["agent:task:cancel"] = {"status": "queued"}
    fake_redis.hashes["agent:task:cancel-ping"] = {"status": "running"}
    assert redis_store.active() == []
    assert redis_store.counts() == {"queued": 0, "running": 0, "total": 0}


def test_redis_delete_and_status_missing(redis_store):
    assert redis_store.status_of("never-existed") is None
    assert redis_store.delete("never-existed") is None
    assert redis_store.cancel_flag("never-existed") is None


def test_redis_ping_and_publish(fake_redis):
    store = RedisTaskStore("redis://x/0", "agent:task:", 600, client=fake_redis)
    assert store.ping() is True, "探活要写+读+删全通，只 PING 会被只读副本骗过"
    store.publish_cancel("sess-x")
    assert fake_redis.published == [("agent:task:cancel", "sess-x")]


def test_router_picks_redis_when_healthy(monkeypatch):
    """探活通过 → 真的用 Redis（不能永远停在 memory，否则配置改了没反应）。"""
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "redis")
    _patched_build_store(monkeypatch, FakeRedis())

    router = TaskStoreRouter()
    assert router.backend == "redis"
    router.save("s1", {"session_id": "s1", "status": "running"})
    assert router.counts()["running"] == 1, "counts 要同时统计 queued 与 running"


def test_build_store_warns_on_unknown_backend(monkeypatch):
    """拼错配置名要明确告警，而不是悄悄按 memory 跑（用户以为自己开了 Redis）。"""
    monkeypatch.setattr(config, "TASK_STATE_BACKEND", "rediis")
    assert isinstance(build_store(), MemoryTaskStore)


def test_redis_active_skips_corrupted_hashes(redis_store, fake_redis):
    """残留的空 Hash（过期键、并发删除）不能让 /health 报错，直接跳过。"""
    fake_redis.hashes["agent:task:ghost"] = {}
    assert redis_store.active() == []


def test_router_degrades_on_read_stats_and_cancel_errors(monkeypatch):
    """读 / 统计 / 快照 / 删除 / 取消 五条路径的降级分支都要能用（否则线上才发现就不灵）。"""
    monkeypatch.setattr(config, "TASK_STATE_HEALTH_CHECK_SECONDS", 9999)
    router = TaskStoreRouter()

    # 每个操作单独注入一次故障，因为第一次就会把 active 降级成 memory
    for _label, call, check in (
        ("status_of", lambda r: r.status_of("s1"), lambda: router.status_of("s1")),
        ("delete", lambda r: r.delete("s1"), lambda: router.status_of("s1")),
    ):
        router._active = _ExplodingStore()
        call(router)
        check()

    router._active = _ExplodingStore()
    assert router.counts() == {"queued": 0, "running": 0, "total": 0}

    router._active = _ExplodingStore()
    assert router.active() == []

    router._active = _ExplodingStore()
    router.mark_cancelled("s1", "user")
    assert router.cancel_flag("s1") is None, "memory 里没有该会话时不该凭空造出取消记录（本地无在途任务可停）"

    # 有本地记录时才必须落 memory：否则本进程的 should_stop 兜底会失效
    router._active = _ExplodingStore()
    router.save("s2", {"session_id": "s2", "status": "running"})  # 降级后落 memory
    router._active = _ExplodingStore()
    router.mark_cancelled("s2", "user")
    assert router.cancel_flag("s2") == "user"


def test_subscribe_thread_survives_pubsub_creation_failure(fake_redis):
    """pubsub() 本身抛异常时线程要干净退出（finally 里的 close 不能踩 UnboundLocalError）。"""
    fake_redis.pubsub_error = ConnectionError("cannot open socket")
    store = RedisTaskStore("redis://x/0", "agent:task:", 600, client=fake_redis)

    store.subscribe_cancel(lambda sid: None)  # 不抛到调用方
    time.sleep(0.3)  # 给线程走到 except/finally 的时间
    assert store.backend == "redis", "订阅失败不影响后端选择"


def test_router_forwards_subscribe_to_active_backend(fake_redis):
    """路由器必须能把订阅转发给当前后端（tasks.py 是在路由器上发起订阅的）。

    回归背景（2026-10-16 真机验证）：路由器漏了这个方法，`/health` 直接 500，
    而当时单测里的替身自带同名方法，把缺陷完美遮住了。
    """
    fake_redis.pubsub_messages = [{"type": "message", "data": "sess-fwd"}]
    router = TaskStoreRouter()
    router._active = RedisTaskStore("redis://x/0", "agent:task:", 600, client=fake_redis)

    done = threading.Event()
    router.subscribe_cancel(lambda sid: done.set())

    assert done.wait(timeout=5), "路由器应把订阅真正转给后端（并收到广播）"
    assert router.backend == "redis", "订阅本身不该改变后端选择"


def test_router_degrades_when_subscribe_fails(monkeypatch):
    """订阅失败时降级为 memory 并给出可用的订阅（空操作），而不是把异常抛给 tasks.py。"""
    monkeypatch.setattr(config, "TASK_STATE_HEALTH_CHECK_SECONDS", 9999)
    router = TaskStoreRouter()
    router._active = _ExplodingStore()

    router.subscribe_cancel(lambda sid: None)  # 不抛异常即可（随后应已降级）

    assert router.backend == "memory"
