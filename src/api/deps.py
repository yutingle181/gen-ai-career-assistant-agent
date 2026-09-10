"""依赖层：会话存储、限流、检索配置构造。"""

from __future__ import annotations

import asyncio
import threading
import uuid

from fastapi import Header, HTTPException

from .. import config
from ..rag.pipeline import RetrievalConfig
from ..rag.registry import get_registry
from ..session import SessionManager

# 进程内会话表（SessionManager 持有 LangChain 消息对象，不入库）
_SESSIONS: dict[str, SessionManager] = {}
_LOCK = threading.Lock()

# 并发限流（JD 职责 9：限流）
MAX_CONCURRENCY = 4
_SEMAPHORE: asyncio.Semaphore | None = None


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def put_session(session_id: str, manager: SessionManager) -> None:
    with _LOCK:
        _SESSIONS[session_id] = manager


def get_session_manager(session_id: str) -> SessionManager | None:
    with _LOCK:
        return _SESSIONS.get(session_id)


def drop_session(session_id: str) -> None:
    with _LOCK:
        _SESSIONS.pop(session_id, None)


def session_count() -> int:
    with _LOCK:
        return len(_SESSIONS)


def get_semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
    return _SEMAPHORE


def build_retrieval_cfg(
    top_k: int | None = None,
    reranker: str | None = None,
    use_bm25: bool | None = None,
    use_vector: bool | None = None,
) -> RetrievalConfig:
    """按需覆盖默认检索配置。"""
    data: dict = {}
    if top_k is not None:
        data["top_k"] = top_k
    if reranker is not None:
        data["reranker"] = reranker
    if use_bm25 is not None:
        data["use_bm25"] = use_bm25
    if use_vector is not None:
        data["use_vector"] = use_vector
    return RetrievalConfig(**data)


def resolve_kb(kb_name: str | None) -> str | None:
    """解析知识库名：未指定时回退到默认库。"""
    registry = get_registry()
    if kb_name and registry.get(kb_name):
        return kb_name
    if kb_name:
        return kb_name  # 允许用户指定一个尚未建库的库名，由上层给提示
    pipeline = registry.ensure_default()
    return pipeline.name


def verify_api_key(api_key: str = Header(None, alias="X-API-Key")) -> None:
    """业务接口鉴权依赖。

    - `config.API_KEY` 为空：放行（兼容演示/教学，零破坏）；
    - 否则校验 `X-API-Key` 请求头，缺失或不匹配返回 401。
    Key 在请求时实时读取，部署时只需在 `.env` 填 `API_KEY` 即生效，无需改码。
    """
    if not config.API_KEY:
        return
    if not api_key or api_key != config.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="缺少或无效的 API Key（请在请求头携带 X-API-Key）。",
        )
