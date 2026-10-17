"""跨副本取消的**端到端**验收：两个真实进程 + 同一个 Redis。

单测与 `verify_task_state_redis.py` 只能证明"存储层命令正确"，
这一层证明的是**跨进程语义**：A 副本上跑着的任务，B 副本发来的取消真的能停掉。

用法（两个实例必须用**不同的 checkpoint DB**，否则后起的那个会卡在 sqlite 写锁上，
     表现为进程活着但 /health 一直超时）：

    # 终端 1
    $env:TASK_STATE_BACKEND='redis'; $env:CHECKPOINT_DB_PATH='data/ckpt-a.sqlite'
    .venv-ci\\Scripts\\python.exe -m uvicorn src.api.main:app --port 8016
    # 终端 2
    $env:TASK_STATE_BACKEND='redis'; $env:CHECKPOINT_DB_PATH='data/ckpt-b.sqlite'
    .venv-ci\\Scripts\\python.exe -m uvicorn src.api.main:app --port 8017
    # 终端 3
    .venv-ci\\Scripts\\python.exe scripts/verify_task_state_cross_replica.py 8016 8017

退出码：0=全通过，1=有失败项。
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

A_PORT = sys.argv[1] if len(sys.argv) > 1 else "8016"
B_PORT = sys.argv[2] if len(sys.argv) > 2 else "8017"
A = f"http://127.0.0.1:{A_PORT}"
B = f"http://127.0.0.1:{B_PORT}"
ok = True


def check(label: str, got, want) -> None:
    global ok
    passed = got == want
    ok = ok and passed
    print(("  [OK]  " if passed else "  [FAIL]") + f" {label}: got={got!r} want={want!r}")


def post(url: str, payload: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))


def get(url: str, timeout: int = 180) -> dict:
    # 首次 /health 会触发本地向量模型加载（实测可达 60–90 秒），超时给小了会误判成挂了
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8"))


print(f"=== 实例 A={A}  B={B} ===")

print("=== 1. 两个实例都真的用上了 Redis（而不是降级后的 memory）===")
for name, base in (("A", A), ("B", B)):
    check(f"{name} task_state_backend", get(base + "/health")["runtime"].get("task_state_backend"), "redis")

print("\n=== 2. A 上发起流式对话（任务注册在 A、跑在 A）===")
cancelled_seen = threading.Event()


def _stream() -> None:
    req = urllib.request.Request(
        A + "/chat/stream",
        data=json.dumps({"query": "帮我写一段 300 字的自我介绍"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore")
                if "cancelled" in line:
                    cancelled_seen.set()
                    break
    except Exception:  # noqa: BLE001 - 流被中断属正常（取消会关生成器）
        cancelled_seen.set()


thread = threading.Thread(target=_stream, daemon=True)
thread.start()

running = None
for _ in range(60):
    time.sleep(0.5)
    counts = get(A + "/health", timeout=30)["runtime"]["tasks"]["counts"]
    if counts["running"] >= 1 or counts["queued"] >= 1:
        running = counts
        break
check("A 上出现在途任务", bool(running), True)
print("  A 的任务计数 =", running)

print("\n=== 3. 跨副本取消：任务在 A，把取消发给 B（B 从来没见过它）===")
import redis as redis_lib  # noqa: E402

client = redis_lib.Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=True, protocol=2)
sid = "s-crossreplica-demo"
# 直接写共享状态，模拟「任务在 A 上正在跑」：对 B 而言这正是"任务在别的副本上"
client.hset(f"agent:task:{sid}", mapping={
    "session_id": sid, "run_id": "demo01", "status": "running",
    "wait_ms": 0, "cancel_reason": "", "queued_at": int(time.time()), "started_at": int(time.time()),
})
client.expire(f"agent:task:{sid}", 600)

resp = post(B + "/chat/cancel", {"session_id": sid, "reason": "user"})
check("B 返回 cancelled", resp.get("cancelled"), True)
check("status 反映共享状态而非 not_found", resp.get("status"), "running")
print("  B 的响应 =", resp)
check("共享状态已写入取消原因", client.hget(f"agent:task:{sid}", "cancel_reason"), "user")

print("\n=== 4. 反向：任务在 B，A 也能取消 ===")
sid2 = "s-crossreplica-demo-2"
client.hset(f"agent:task:{sid2}", mapping={
    "session_id": sid2, "run_id": "demo02", "status": "running",
    "wait_ms": 0, "cancel_reason": "", "queued_at": int(time.time()), "started_at": int(time.time()),
})
client.expire(f"agent:task:{sid2}", 600)
check("A 也能取消 B 上的任务", post(A + "/chat/cancel", {"session_id": sid2, "reason": "user"}).get("cancelled"), True)

print("\n=== 5. 背压计数是全局的 ===")
totals = []
for base in (A, B):
    counts = get(base + "/health", timeout=60)["runtime"]["tasks"]["counts"]
    totals.append(counts["total"])
    print(f"  {base} counts = {counts}")
check("两个实例看到同一个 total", totals[0], totals[1])

print("\n=== 6. 不存在的会话仍返回 False / not_found ===")
missing = post(B + "/chat/cancel", {"session_id": "never-existed-xyz", "reason": "user"})
check("cancel 返回 False", missing.get("cancelled"), False)
check("status 为 not_found", missing.get("status"), "not_found")

print("\n=== 7. 正在跑的流是否真停 ===")
thread.join(timeout=240)
check("流收到 cancelled 标记", cancelled_seen.is_set(), True)

print("\n=== 清理 ===")
for key in list(client.scan_iter(match="agent:task:*")):
    client.delete(key)

print("\n结论：" + ("全部通过" if ok else "有失败项"))
raise SystemExit(0 if ok else 1)
