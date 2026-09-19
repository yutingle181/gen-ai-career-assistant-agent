"""健康检查：模型连通性、Embedding 可用性、配置快照（已脱敏）。"""

from __future__ import annotations

from fastapi import APIRouter

from ... import config, faults, metrics, tasks
from ...embeddings import check_embedding_health
from ...graph.checkpoint import backend_name
from ...llm import check_llm_health
from ...mcp import snapshot as mcp_snapshot
from ...rag.registry import get_registry
from ...tools import get_breaker
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
        runtime={
            "checkpoint": backend_name(),
            "tool_circuit": get_breaker().snapshot(),
            "session_resume": config.ENABLE_SESSION_RESUME,
            "slot_memory": config.ENABLE_SLOT_MEMORY,
            "tool_calling": config.ENABLE_TOOL_CALLING,
            # 故障注入状态也要可见：否则「这次为什么全在失败」会变成没人说得清的悬案
            "fault_injection": faults.describe(),
            # 在途任务（排队 / 执行中）：背压与取消都必须能被看见，
            # 否则「为什么一直转圈」「点了停止到底停没停」只能靠猜。
            "tasks": tasks.snapshot(),
            "api_max_queue": config.API_MAX_QUEUE,
            # MCP 外部工具状态（不含 command / env —— 那些可能含凭据）
            "mcp": mcp_snapshot(),
        },
        metrics=metrics.snapshot(),
    )


@router.get("/config")
async def show_config():
    return config.describe()
