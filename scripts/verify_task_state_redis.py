"""真 Redis 验证脚本：对 store 层的每条命令对着**真 Redis**跑一遍。

为什么必须有这一层（假客户端不够）：
假客户端是按「我以为的 API」写的。若签名/语义记错——比如 `hset(mapping=)`、
`scan_iter` 的 match 行为、pipeline 的 expire——假客户端会**跟着一起错**，
于是单测全绿而线上直接崩。2026-10-16 就是这样抓到的：redis-py 8.x 默认发
`HELLO` 切 RESP3，而 Redis 5.x 不认这个命令，假客户端完全模拟不出来。

用法（需要一个可连的 Redis；本机 Windows 版实测可用）：
    D:\\env\\redis\\redis-server.exe D:\\env\\redis\\redis.windows.conf --dir D:\\env\\redis --save ""
    .venv-ci\\Scripts\\python.exe scripts\\verify_task_state_redis.py [redis_url]

退出码：0=全通过，1=有失败项。
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.tasks_store import RedisTaskStore  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "redis://127.0.0.1:6379/0"
PREFIX = "agent:verify:"

store = RedisTaskStore(URL, PREFIX, ttl_seconds=60)
ok = True


def check(label: str, got, want) -> None:
    global ok
    passed = got == want
    ok = ok and passed
    print(("  [OK]  " if passed else "  [FAIL]") + f" {label}: got={got!r} want={want!r}")


print(f"=== 连接目标：{URL} ===")

print("=== 1. 探活（写+读+删，不只 PING：只读副本会让 PING 骗人）===")
check("ping", store.ping(), True)

print("=== 2. 写入 / 读取状态 ===")
store.save("s1", {"session_id": "s1", "run_id": "r1", "status": "queued",
                  "wait_ms": 0, "cancel_reason": "", "queued_at": 100, "started_at": None})
check("status_of", store.status_of("s1"), "queued")
store.save("s2", {"session_id": "s2", "run_id": "r2", "status": "running",
                  "wait_ms": 120, "queued_at": 90, "started_at": 210})
check("status_of s2", store.status_of("s2"), "running")
check("不存在返回 None", store.status_of("nope"), None)

print("=== 3. 全局计数（跨副本核心：扫在途键按 status 汇总）===")
counts = store.counts()
check("queued", counts["queued"], 1)
check("running", counts["running"], 1)
check("total", counts["total"], 2)

print("=== 4. 取消状态位（空串必须归一化成 None，与内存实现同语义）===")
check("未取消时 cancel_flag", store.cancel_flag("s1"), None)
store.mark_cancelled("s1", "user")
check("取消后 cancel_flag", store.cancel_flag("s1"), "user")

print("=== 5. 终态不计入在途 ===")
store.save("s3", {"session_id": "s3", "status": "done"})
check("done 不算在途", store.counts()["total"], 2)
check("active 只含在途", sorted(i["session_id"] for i in store.active()), ["s1", "s2"])

print("=== 6. 广播（无订阅者也不该抛：Pub/Sub 不保证送达，靠状态位兜底）===")
store.publish_cancel("s1")

print("=== 7. 删除 ===")
store.delete("s1")
check("删除后 status", store.status_of("s1"), None)
check("删除后 total", store.counts()["total"], 1)

print("=== 8. TTL（终态靠 TTL 自然清理，不留垃圾）===")
store.save("s4", {"session_id": "s4", "status": "queued"})
ttl = store._redis.ttl(PREFIX + "s4")
check("TTL 已设置", 0 < ttl <= 60, True)

print("=== 9. 脏数据不能打挂 /health ===")
store._redis.hset(PREFIX + "s5", mapping={
    "session_id": "s5", "status": "running", "queued_at": "不是数字"})
item = next((i for i in store.active() if i["session_id"] == "s5"), None)
check("坏数字按 0 处理", item and item["queued_at"], 0)

print("=== 清理 ===")
for key in list(store._redis.scan_iter(match=PREFIX + "*")):
    store._redis.delete(key)

print("\n结论：" + ("全部通过" if ok else "有失败项"))
raise SystemExit(0 if ok else 1)
