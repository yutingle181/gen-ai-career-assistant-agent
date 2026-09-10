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

# ============ 阶段 1：构建依赖 ============
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 依赖装进独立 venv：runtime 阶段整体拷贝，不污染系统 site-packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
# 用锁定版本安装：镜像内依赖完全可复现，不受上游发版影响
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

# ============ 阶段 2：运行镜像 ============
FROM python:3.12-slim

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

# 默认启动 API 服务；compose 中 web 服务用 command 覆盖
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
