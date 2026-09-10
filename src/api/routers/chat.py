"""对话路由：同步对话与 SSE 流式对话。"""

from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from ...cache import Timer, get_cost_tracker
from ...graph.workflow import route_only
from ...logging_setup import get_logger
from ...safety import check_input, safe_output
from ...session import SessionManager
from ...state import MODE_LABELS
from .. import db
from ..deps import build_retrieval_cfg, get_semaphore, get_session_manager, new_session_id, put_session, resolve_kb
from ..schemas import ChatRequest, ChatResponse, FinishRequest

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


def _resolve_mode(req: ChatRequest) -> str:
    if req.mode:
        return req.mode
    return route_only(req.query).get("mode", "qa")


def _ensure_session(req: ChatRequest) -> tuple[str, SessionManager, bool]:
    """获取或新建会话，返回 (session_id, manager, is_new)。"""
    if req.session_id:
        manager = get_session_manager(req.session_id)
        if manager is not None:
            return req.session_id, manager, False

    mode = _resolve_mode(req)
    kb_name = resolve_kb(req.kb_name) if mode == "knowledge" else req.kb_name
    cfg = build_retrieval_cfg(req.top_k, req.reranker, req.use_bm25, req.use_vector)
    manager = SessionManager(mode, kb_name=kb_name, retrieval_cfg=cfg, user_context=req.user_context)
    sid = req.session_id or new_session_id()
    put_session(sid, manager)
    db.create_session(sid, mode, kb_name or "")
    logger.info("新建会话 | id=%s mode=%s", sid, mode)
    return sid, manager, True


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """一轮对话（非流式）。"""
    ok, reason = check_input(req.query)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    sem = get_semaphore()
    async with sem:  # 限流
        sid, manager, is_new = _ensure_session(req)
        db.add_message(sid, "user", req.query)

        with Timer() as t:
            reply = await asyncio.to_thread(
                manager.start if is_new else manager.step, req.query
            )
        db.add_message(sid, "assistant", reply)

        tracker = get_cost_tracker()
        db.log_cost(sid, tracker.prompt_tokens, tracker.completion_tokens, t.ms)

        return ChatResponse(
            session_id=sid,
            mode=manager.mode,
            mode_label=MODE_LABELS.get(manager.mode, manager.mode),
            message=reply,
            citations=manager.citations,
            artifact=manager.artifact_path,
            finished=manager.finished,
            requires_confirmation=manager.requires_confirmation,
            awaiting_confirmation=manager.awaiting_confirmation,
            draft_artifact=manager.draft_path,
        )


@router.post("/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式对话：先推送 mode 事件，再逐段推送文本。"""
    import json

    ok, reason = check_input(req.query)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    sem = get_semaphore()
    await sem.acquire()

    sid, manager, is_new = _ensure_session(req)
    db.add_message(sid, "user", req.query)

    async def gen():
        import json

        from langchain_core.messages import HumanMessage

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def produce() -> None:
            """在独立线程里跑阻塞的同步流式生成，逐段推给事件循环。"""
            try:
                for piece in manager.agent.respond_stream(manager._trim_history()):
                    asyncio.run_coroutine_threadsafe(queue.put(piece), loop).result()
            except Exception as exc:  # noqa: BLE001
                err = exc if isinstance(exc, RuntimeError) else RuntimeError(f"（模型调用失败：{exc}）")
                asyncio.run_coroutine_threadsafe(queue.put(err), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

        producer = threading.Thread(target=produce, daemon=True)

        try:
            payload = {"type": "session", "session_id": sid, "mode": manager.mode}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # 首轮需要把「用户输入 + 外部资料」一起入栈
            if is_new:
                context = await asyncio.to_thread(manager.agent.prepare, req.query)
                manager.history.append(HumanMessage(content=f"{req.query}{context}"))
                manager.started = True
                manager.record.append(f"**用户**：{req.query}\n")
            else:
                manager.record.append(f"\n**用户**：{req.query}\n")
                manager.history.append(HumanMessage(content=req.query))

            buffer: list[str] = []
            producer.start()
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    logger.error("流式生成失败：%s", item)
                    message = str(item)
                    buffer.append(message)
                    yield f"data: {json.dumps({'type': 'delta', 'text': message}, ensure_ascii=False)}\n\n"
                    break
                buffer.append(item)
                yield f"data: {json.dumps({'type': 'delta', 'text': item}, ensure_ascii=False)}\n\n"

            text = safe_output("".join(buffer))
            from langchain_core.messages import AIMessage

            manager.history.append(AIMessage(content=text))
            manager.record.append(f"\n**助手**：{text}\n")
            if manager.agent.one_shot:
                manager.finish()

            payload = {
                "type": "done",
                "citations": manager.citations,
                "artifact": manager.artifact_path,
                "finished": manager.finished,
                "requires_confirmation": manager.requires_confirmation,
                "awaiting_confirmation": manager.awaiting_confirmation,
                "draft_artifact": manager.draft_path,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            sem.release()

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/finish", response_model=ChatResponse)
async def finish(req: FinishRequest) -> ChatResponse:
    """结束会话并导出 Markdown。

    需人工确认的场景这里只产出**草稿**（数据库不标记完成），
    需再调用 `POST /chat/confirm` 才落盘定稿。
    """
    manager = get_session_manager(req.session_id)
    if manager is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    path = manager.finish()
    # 草稿态不能标记完成：只有真正定稿才写库，避免「草稿被当成已完结」
    if manager.finished:
        db.mark_finished(req.session_id, path or "")

    awaiting = manager.awaiting_confirmation
    return ChatResponse(
        session_id=req.session_id,
        mode=manager.mode,
        mode_label=MODE_LABELS.get(manager.mode, manager.mode),
        message=f"草稿已生成，请审阅后确认定稿：{path}" if awaiting else f"会话已导出：{path}",
        citations=manager.citations,
        artifact=manager.artifact_path,  # 草稿态下为 None，草稿走 draft_artifact
        finished=manager.finished,
        requires_confirmation=manager.requires_confirmation,
        awaiting_confirmation=awaiting,
        draft_artifact=manager.draft_path,
    )


@router.post("/confirm", response_model=ChatResponse)
async def confirm(req: FinishRequest) -> ChatResponse:
    """人工确认：把草稿定稿为最终产物（人机协同收口）。"""
    manager = get_session_manager(req.session_id)
    if manager is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    path = manager.confirm()
    db.mark_finished(req.session_id, path or "")
    logger.info("人工确认定稿 | session=%s | %s", req.session_id, path)
    return ChatResponse(
        session_id=req.session_id,
        mode=manager.mode,
        mode_label=MODE_LABELS.get(manager.mode, manager.mode),
        message=f"已人工确认并定稿：{path}",
        citations=manager.citations,
        artifact=path,
        finished=True,
        requires_confirmation=False,
        awaiting_confirmation=False,
        draft_artifact=manager.draft_path,
    )
