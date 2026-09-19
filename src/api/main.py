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
    yield


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
        allow_origins=["*"],
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

    @app.get("/")
    async def root():
        return {
            "name": "GenAI Career Assistant API",
            "docs": "/docs",
            "health": "/health",
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

    return app


app = create_app()
