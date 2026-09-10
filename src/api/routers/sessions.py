"""会话路由：列表、历史、成本统计。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...cache import get_cost_tracker
from ...logging_setup import get_logger
from ...state import MODE_LABELS
from .. import db
from ..schemas import CostSummary

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = get_logger(__name__)


@router.get("")
async def list_sessions(limit: int = 50):
    rows = db.list_sessions(limit)
    return [
        {
            "id": r.id,
            "mode": r.mode,
            "mode_label": MODE_LABELS.get(r.mode, r.mode),
            "kb_name": r.kb_name,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
            "finished": r.finished,
            "artifact": r.artifact,
        }
        for r in rows
    ]


@router.get("/{session_id}")
async def get_session_detail(session_id: str):
    row = db.get_session(session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = db.list_messages(session_id)
    return {
        "id": row.id,
        "mode": row.mode,
        "kb_name": row.kb_name,
        "finished": row.finished,
        "artifact": row.artifact,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    from ..deps import drop_session

    drop_session(session_id)
    return {"deleted": session_id}


@router.get("/cost/summary", response_model=CostSummary)
async def cost_summary():
    """当前进程的 token 与成本统计。"""
    s = get_cost_tracker().summary()
    return CostSummary(
        calls=s["调用次数"],
        prompt_tokens=s["输入 tokens"],
        completion_tokens=s["输出 tokens"],
        total_tokens=s["合计 tokens"],
        cost_usd=s["估算费用(USD)"],
        latency_ms=s["累计耗时(ms)"],
    )
