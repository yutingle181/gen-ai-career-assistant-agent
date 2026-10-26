# GenAI Career Assistant —— 容器镜像（CPU / Python 3.12，无 GPU、无 torch）
#
# 构建：  docker build -t genai-career-assistant .
# 编排：  见 docker-compose.yml（docker compose up --build）
#
# 设计要点：
# - 多阶段构建：依赖装进 builder 阶段的独立 venv，最终镜像只拷贝 venv，
#   不带 pip 缓存与任何构建期残留，镜像更小、攻击面更小。
# - 非 root 运行：以 uid 10001 的 appuser 启动，容器逃逸也不会直接拿到 root。
# - 数据与密钥均不进镜像：./data 以 volume 挂载，密钥由 compose 的 env_file 注入。
# - 供应链固定：基镜像按 digest 钉定（tag 只是别名，会被上游覆盖；Dependabot 的 docker
#   生态负责跟进新 digest），依赖安装开启 --require-hashes，只接受锁文件里登记的产物。

# ============ 阶段 1：构建依赖 ============
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 依赖装进独立 venv：runtime 阶段整体拷贝，不污染系统 site-packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
# 用锁定版本安装：镜像内依赖完全可复现，不受上游发版影响
# --require-hashes：逐个校验下载产物的 sha256，防止「同名同版本但内容被替换」的投毒/中间人
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock.txt

# ============ 阶段 2：运行镜像 ============
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

LABEL org.opencontainers.image.title="GenAI Career Assistant" \
      org.opencontainers.image.description="RAG + Agent 职业助手（CPU / 无 torch）"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 只拷贝 venv：不含 pip 缓存、构建工具与任何中间产物
COPY --from=builder /opt/venv /opt/venv

# 再拷源码（数据与密钥不进镜像）
COPY src ./src
COPY app.py .

# 非 root 运行：创建系统用户，并把工作目录与数据 / 产物目录交其所有
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p data/knowledge data/index data/cache data/eval Agent_output \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

# 监听端口做成可覆盖：默认 8000（本机 / compose / 生产一致），
# 部署到 Hugging Face Spaces 时设 PORT=7860（平台约定）即可，**不用改 Dockerfile** ✓✓。
# 注意不能简单写成 CMD [... , "--port", "8000"] 硬编码 —— 那样 HF 上会起不来 ✗。
ENV PORT=8000

# 容器级探活（compose 里也会定义一份，这里让裸 docker run 也有）。
# 用 /healthz 而不是 /health：/health 会真实调一次 LLM + Embedding，
# 每 30 秒烧一次 token，且模型超时会直接判定容器不健康 → 被反复重启，越探测越坏。
#
# ⚠️ 为什么探活是独立脚本而不是写在 Dockerfile 里（2026-10-18 真机构建踩到 ✨✓）：
#   Dockerfile 里**没有** `# syntax=docker/dockerfile:1` 这行时，`CMD-SHELL` 是**非法指令**，
#   经典 builder 直接报
#     Unknown type "CMD-SHELL" in HEALTHCHECK (try CMD)
#   → 镜像构建失败 ✗✗
#   而 exec 形式（CMD ["python","-c","..."])）不展开 ${PORT} ✨✓
#   → HF Spaces 的 7860 端口探不到，探活永远失败 ✓✗
#   用 sh -c 包一层虽能读环境变量，但引号要嵌四层（JSON + shell + Python + URL）✨✓ 极易错 ✓✗
#   → 所以独立成 scripts/healthcheck.py：可读、可单测、Dockerfile 里就一行 ✓✓
COPY scripts/healthcheck.py /usr/local/bin/healthcheck.py
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "/usr/local/bin/healthcheck.py"]

# SIGTERM 后给 30 秒收尾（SSE 长连接与在途任务需要），compose 用 stop_grace_period 对齐
STOPSIGNAL SIGTERM

# 默认启动 API 服务；compose 中 web 服务用 command 覆盖。
# 用 sh -c + exec 而不是 exec 数组形式：数组形式不展开 ${PORT} ✗；
# exec 让 uvicorn 替换掉 sh 成为 PID 1，**信号仍能直达** ✓（否则优雅退出失效 ✗✗）。
CMD ["sh", "-c", "exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
