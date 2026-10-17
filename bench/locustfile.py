"""压测场景：把"这个数字到底测的是什么"写在场景名字上。

为什么强调这一点：压测最容易被拿来误导——一个 3000 QPS 的数字，
如果没说清压的是"框架空转"还是"真实业务路径"，它就没有任何意义。
所以本文件里每个场景都标注了**是否调用 LLM**，报告里也会重复一遍。

三个场景（成本从低到高）：

| 场景 | 打的是什么 | 调 LLM？ | 数字代表什么 |
| --- | --- | --- | --- |
| `baseline` | `GET /` | 否 | 框架空转上限：FastAPI + 中间件（限流/CORS/OTel 兜底） |
| `retrieval` | `POST /knowledge/{kb}/search` | 否（本地 BGE） | 混合检索真实拐点：BM25 + 向量 + RRF + rerank 全链路 |
| `sse_ttft` | `POST /chat/stream` | **是**（烧 token） | 首事件时延 TTFT；默认 weight=0 不参与，要压加 `-IncludeChat` |

`baseline` 的数字最容易被人拿去吹"QPS 破千"，所以这里特别说明：
它只证明"框架没成为瓶颈"，**不能**证明"系统能扛住真实流量"——
真实路径里 embedding 与 LLM 才是耗时主体。

用法（本地，不需要 Docker）：
    .venv-ci\\Scripts\\python.exe -m locust -f bench\\locustfile.py --headless \\
        --host http://127.0.0.1:8000 -u 10 -r 2 -t 30s

更推荐用 bench/run_bench.ps1（自动预热 + 阶梯并发 + 画曲线）；直接调 locust 时
**必须加 `--headless`**，否则它会尝试起 Web UI 并阻塞。

注意压测前先起服务：
    .venv-ci\\Scripts\\python.exe -m uvicorn src.api.main:app --port 8000
"""

from __future__ import annotations

import os
import time

from locust import HttpUser, between, events, task

#: 知识库名（samples 随仓库入库，任何人可复现；data1 来自上传目录，不可复现）
KB = os.getenv("BENCH_KB", "samples")

#: SSE 场景的 query；刻意选短问题，减少 token 消耗（TTFT 与问题长度弱相关）
SSE_QUERY = os.getenv("BENCH_SSE_QUERY", "你好")

#: 只跑某一个场景（run_bench.ps1 每轮设一次）。
#:
#: 为什么用 weight 而不是 `--tags` / `--class-picker`：实测 locust 2.46 上
#:   · `--class-picker BaselineUser` 与 `--class-picker 'A|B'` 都报 "Unknown User(s)"；
#:   · `--tags 'baseline or retrieval'` 和 `--tags 'tag:retrieval'` 都把**所有**类过滤掉，
#:     结果是 0 请求（locust 的 tag 匹配的是 **Task 级** `@tag`，不是 User 类上的 `tags`）。
#: 而 `weight = 0` 是被实测确认有效的"排除某个类"的方式 ✓ —— locust 不会为权重 0 的类建用户 ✓。
_ONLY = os.getenv("BENCH_SCENARIO", "").strip()


def _weight(name: str, default: int) -> int:
    """未指定场景时按默认权重；指定了场景则只有它有权重。"""
    return default if (not _ONLY or _ONLY == name) else 0


class BaselineUser(HttpUser):
    """框架空转基线：`GET /`。不调模型、不碰数据库（只返回静态 JSON）。"""

    weight = _weight("baseline", 3)
    wait_time = between(0, 0.05)

    @task
    def ping_root(self):
        self.client.get("/", name="GET /")


class RetrievalUser(HttpUser):
    """检索真实路径：BM25 + 本地向量 + RRF 融合 + rerank。

    不调 LLM（本地 BGE 编码），但**吃 CPU**：embedding 是这条链路里最重的部分，
    所以并发上去最先拐弯的就是它——这正是压测想找的东西。
    """

    weight = _weight("retrieval", 2)
    wait_time = between(0.5, 1.5)

    @task
    def search(self):
        self.client.post(
            f"/knowledge/{KB}/search",
            json={"query": "Python 的 FastAPI 怎么用", "top_k": 5},
            name=f"POST /knowledge/{KB}/search",
        )


class SseTtftUser(HttpUser):
    """SSE 首事件时延（TTFT）。**会调用 LLM，消耗 token，默认不参与压测。**

    需要显式开启：`.venv-ci\\Scripts\\python.exe -m locust ... --class-picker SseTtftUser`
    或用 `bench/run_bench.ps1 -IncludeChat`。
    单请求单线程（`wait_time` 很长），因为关注的是**首事件**而不是吞吐。
    """

    weight = _weight("llm", 0)  # 默认 0：默认不压 LLM 路径（会烧 token）
    wait_time = between(3, 6)

    @task
    def first_token(self):
        started = time.perf_counter()
        with self.client.post(
            "/chat/stream",
            json={"query": SSE_QUERY},
            name="POST /chat/stream (TTFT)",
            stream=True,
            catch_response=True,
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8", "ignore")
                if text.startswith("data:") and "chunk" in text:
                    # 首个 chunk 到达 = 用户看到第一个字
                    ttft_ms = (time.perf_counter() - started) * 1000
                    resp.request_meta = {"ttft_ms": ttft_ms}
                    events.request.fire(
                        request_type="TTFT",
                        name="POST /chat/stream (first chunk)",
                        response_time=ttft_ms,
                        response_length=0,
                        exception=None,
                        context={},
                    )
                    break
            else:
                resp.failure("SSE 未收到首个 chunk")


@events.quitting.add_listener
def _print_summary(environment, **_kwargs):  # noqa: ANN001
    """结束时把关键结论打印到控制台，方便 CI/终端直接看。"""
    stats = environment.stats
    print("\n===== 压测小结 =====")
    print(f"总请求 {stats.total.num_requests}  失败 {stats.total.num_failures}")
    for entry in stats.entries.values():
        pct = entry.get_response_time_percentile(0.95)
        print(
            f"  {entry.name:<44} n={entry.num_requests:<6} "
            f"P50={entry.median_response_time}ms P95={pct}ms 失败={entry.num_failures}"
        )
    print("提示：baseline 的数字只说明框架不是瓶颈，不代表真实业务吞吐。")
