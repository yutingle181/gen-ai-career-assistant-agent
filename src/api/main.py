"""FastAPI 应用装配。"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .. import __version__, config
from ..logging_setup import get_logger
from ..telemetry import setup_telemetry
from . import db
from .deps import verify_api_key
from .routers import chat, health, knowledge, sessions

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """进程内固定时间窗限流：按客户端 IP 计数，超过 API_RATE_LIMIT 次/分钟返回 429。

    单进程 uvicorn 场景足够、零依赖；多副本时后续可换 Redis。限流上限实时读取
    config.API_RATE_LIMIT（<=0 表示关闭），白名单路径（探活/文档）不计数。
    """

    _WHITELIST_DEFAULT = {"/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, whitelist: set[str] | None = None) -> None:
        super().__init__(app)
        self.whitelist = whitelist or self._WHITELIST_DEFAULT
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        limit = config.API_RATE_LIMIT
        if limit <= 0 or request.url.path in self.whitelist:
            return await call_next(request)
        ident = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(ident, [])
            window[:] = [t for t in window if now - t < 60]
            if len(window) >= limit:
                retry = max(int(60 - (now - window[0])), 1) if window else 60
                return JSONResponse(
                    status_code=429,
                    content={"detail": "请求过于频繁，请稍后再试。"},
                    headers={"Retry-After": str(retry)},
                )
            window.append(now)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    logger.info(
        "API 启动 | model=%s | embedding=%s", config.MODEL_NAME, config.EMBEDDING_PROVIDER
    )
    # 公网暴露风险自查：默认配置适合本地，但直接挂到公网等于把模型额度裸奔出去。
    # 启动时说清，比账单出来才发现强（2026-10-16 上云准备时补的）。
    if not config.auth_enabled():
        logger.warning(
            "API_KEY 未设置：业务接口当前**无需鉴权**即可调用。"
            "本地开发这样最省事，但若把服务暴露到公网，任何人都能消耗你的模型额度。"
        )
    for risk in config.public_exposure_risks():
        logger.warning("公网暴露风险：%s", risk)
    yield
    # ---- 优雅退出的收尾位置（目前只记日志，真正的清理见下方注释）----
    # 为什么不关 SQLite / 不停在途任务：
    #   - SQLite 由 SQLAlchemy 引擎持有，进程退出即释放；显式关闭意义不大，
    #     而在 lifespan 里抛异常会掩盖真正的退出原因；
    #   - 在途任务是「请求正在进行」，容器滚动更新时被 SIGTERM 打断属于正常运维，
    #     uvicorn 会先停止接收新请求、等已有请求收尾；我们要做的是**别把在途任务
    #     留成静默烧 token 的孤儿**（这由 Redis 共享状态 + TTL 兜底，见 tasks_store）。
    # 所以这里只打一行日志，让"重启发生过"在日志里可查。
    logger.info("API 正在退出（优雅关闭已启动）")


def create_app() -> FastAPI:
    setup_telemetry()  # 仅当配置了 OTEL_EXPORTER_OTLP_ENDPOINT 时启用导出

    app = FastAPI(
        title="GenAI Career Assistant API",
        description="LangGraph 多 Agent 职业助手 + 企业知识库 RAG 引擎",
        version=__version__,
        lifespan=lifespan,
    )
    # FastAPI 自动埋点：每个请求生成 server.request span（未启用 OTel 时为 no-op）
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:  # noqa: BLE001
        pass

    db.init_db()  # 幂等建表，避免某些 ASGI 宿主不触发 lifespan 时报错

    app.add_middleware(
        CORSMiddleware,
        # 显式白名单（config.CORS_ORIGINS），不再用通配符：
        # `allow_origins=["*"]` 与 `allow_credentials=True` 是**非法组合**，
        # 浏览器会直接拒掉这类响应——也就是说原来那个配置在跨域场景下根本不生效，
        # 属于"看起来配了、实际没用"的假配置（2026-10-16 修正）。
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ANN201
        logger.error("未处理异常 %s %s | %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    app.include_router(health.router)
    app.include_router(chat.router, dependencies=[Depends(verify_api_key)])
    app.include_router(knowledge.router, dependencies=[Depends(verify_api_key)])
    app.include_router(sessions.router, dependencies=[Depends(verify_api_key)])

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "GenAI Career Assistant API",
            "docs": "/docs",
            "health": "/health",
            "healthz": "/healthz",
            "endpoints": [
                "POST /chat",
                "POST /chat/stream (SSE)",
                "POST /chat/cancel",
                "POST /chat/finish",
                "GET  /knowledge",
                "POST /knowledge/{kb}/ingest",
                "POST /knowledge/{kb}/search",
                "GET  /sessions",
                "GET  /sessions/cost/summary",
            ],
        }

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        """轻量探活（liveness/readiness）：**零外部依赖**。

        为什么必须有一个和 `/health` 分开的端点（2026-10-16 上云准备时发现）：
        - `/health` 会**真实调用一次 LLM 与一次 Embedding**（`check_llm_health`），
          拿它做 compose / nginx / k8s 的健康检查，等于每 30 秒烧一次 token，
          而且 LLM 超时会让探活失败 → 容器被反复重启，**越探测越坏**；
        - 它还需要 Key 才判得出"未配置"，无 Key 时仍返回 200 + degraded，
          探活根本区分不出"活着"和"不可用"。

        所以这里只返回静态信息：能返回 200 就说明进程活着、路由已注册、事件循环没卡死。
        要诊断模型/向量/索引，请用 `/health`（人工点开或低频调用）。
        """
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
