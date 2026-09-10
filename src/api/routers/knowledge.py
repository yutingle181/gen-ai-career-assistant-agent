"""知识库路由：上传建库、列表、删除、检索。"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ... import config
from ...logging_setup import get_logger
from ...rag.pipeline import RetrievalConfig
from ...rag.registry import get_registry
from ..schemas import IngestResponse, KnowledgeInfo, SearchHit, SearchRequest

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
logger = get_logger(__name__)

_UPLOAD_ROOT = config.DATA_DIR / "uploads"


@router.get("", response_model=list[KnowledgeInfo])
async def list_knowledge():
    return [KnowledgeInfo(**s) for s in get_registry().list()]


@router.post("/{kb_name}/ingest", response_model=IngestResponse)
async def ingest(kb_name: str, files: list[UploadFile] = File(...)):
    """上传文档并建库（覆盖式重建索引）。"""
    target_dir = _UPLOAD_ROOT / kb_name
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        dest = target_dir / (f.filename or "unnamed")
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved.append(str(dest))

    if not saved:
        raise HTTPException(status_code=400, detail="未收到任何文件")

    pipeline = get_registry().get_or_create(kb_name)
    try:
        result = pipeline.ingest(saved)
    except Exception as exc:  # noqa: BLE001
        logger.error("建库失败：%s", exc)
        raise HTTPException(status_code=500, detail=f"建库失败：{type(exc).__name__}: {exc}") from exc

    return IngestResponse(
        kb_name=kb_name,
        files=result.files,
        documents=result.documents,
        chunks=result.chunks,
        dim=result.dim,
        elapsed_ms=result.elapsed_ms,
    )


@router.post("/{kb_name}/ingest_dir", response_model=IngestResponse)
async def ingest_dir(kb_name: str, directory: str):
    """从服务器本地目录建库（便于批量导入）。"""
    path = Path(directory)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在：{directory}")
    pipeline = get_registry().get_or_create(kb_name)
    result = pipeline.ingest([str(path)])
    return IngestResponse(
        kb_name=kb_name,
        files=result.files,
        documents=result.documents,
        chunks=result.chunks,
        dim=result.dim,
        elapsed_ms=result.elapsed_ms,
    )


@router.delete("/{kb_name}")
async def delete_knowledge(kb_name: str):
    ok = get_registry().delete(kb_name)
    if not ok:
        raise HTTPException(status_code=404, detail="知识库不存在")
    shutil.rmtree(_UPLOAD_ROOT / kb_name, ignore_errors=True)
    return {"deleted": kb_name}


@router.post("/{kb_name}/search", response_model=list[SearchHit])
async def search(kb_name: str, req: SearchRequest):
    pipeline = get_registry().get(kb_name)
    if pipeline is None or not pipeline.ready:
        raise HTTPException(status_code=404, detail="知识库不存在或尚未建库")
    cfg = RetrievalConfig(top_k=req.top_k, reranker=req.reranker)
    chunks = pipeline.retrieve(req.query, cfg)
    return [
        SearchHit(
            chunk_id=c.chunk_id,
            source=c.source,
            page=c.page,
            score=c.score,
            text=c.text,
        )
        for c in chunks
    ]
