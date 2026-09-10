"""健康检查：模型连通性、Embedding 可用性、配置快照（已脱敏）。"""

from __future__ import annotations

from fastapi import APIRouter

from ... import config
from ...embeddings import check_embedding_health
from ...llm import check_llm_health
from ...rag.registry import get_registry
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    llm_ok, llm_detail = check_llm_health()
    emb_ok, emb_detail = check_embedding_health()
    return HealthResponse(
        status="ok" if (llm_ok and emb_ok) else "degraded",
        llm=llm_ok,
        llm_detail=llm_detail,
        embedding=emb_ok,
        embedding_detail=emb_detail,
        knowledge_bases=get_registry().names(),
        config=config.describe(),
    )


@router.get("/config")
async def show_config():
    return config.describe()
