#!/usr/bin/env python3
"""容器级探活：只验「进程活着 + 事件循环没卡死」，**不调模型、不查索引**。

为什么单独一个文件而不是写在 Dockerfile 的 HEALTHCHECK 里：
    1. Dockerfile 里没有 `# syntax=docker/dockerfile:1` 时，`CMD-SHELL` 是**非法指令**
       （经典 builder 直接报 `Unknown type "CMD-SHELL" in HEALTHCHECK` ✗✗）。
       而 exec 形式（CMD ["python","-c","..."]）不展开 ${PORT} ✨✓
       → HF Spaces 的 7860 端口探不到，探活永远失败 ✓✗。
    2. 用 sh -c 包一层能读环境变量，但引号要嵌四层（JSON 引号 + shell 引号 +
       Python 引号 + URL 里的引号）✨✓ 极易写错，且报错信息很难读 ✓✗。
    3. 独立脚本：可读、可单测、Dockerfile 里就一行 ✓✓

为什么用 /healthz 而不是 /health：
    /health 会真实调一次 LLM + Embedding —— 每 30 秒烧一次 token，
    且模型超时会直接判定容器不健康 → 被反复重启，**越探测越坏** ✓✗。
    /healthz 只回静态信息，返回 200 就说明进程活着、事件循环没卡死 ✓✓
    （要诊断模型 / 向量 / 索引请手动 curl /health ✓✓）。

退出码：0 = 健康；非 0 = 不健康（Docker 据此判 unhealthy ✨✓）。
"""
from __future__ import annotations

import os
import sys
import urllib.request

TIMEOUT_SECONDS = 5


def main() -> int:
    # 端口可覆盖：默认 8000（本机 / compose / 自建服务器 ✓✓），
    # HF Spaces 等平台会注入自己的端口（7860 ✓✓）。
    # 做成环境变量而不是硬编码，是因为同一个镜像要跑在不同平台上 ✨✓
    port = os.environ.get("PORT", "8000")
    url = f"http://localhost:{port}/healthz"

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp:
            if resp.status == 200:
                return 0
            print(f"healthcheck: {url} 返回 {resp.status}", file=sys.stderr)
            return 1
    except Exception as exc:  # noqa: BLE001 - 探活要把任何异常都变成"不健康"
        print(f"healthcheck: {url} 失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
