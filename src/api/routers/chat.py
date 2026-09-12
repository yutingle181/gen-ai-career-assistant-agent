"""对话路由：同步对话与 SSE 流式对话。"""

from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

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

# 静默心跳间隔（秒）：远小于网关 120s 读超时，保证长任务不会被中间层掐断
KEEPALIVE_SECONDS = 15.0


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


        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def produce() -> None:
            """在独立线程里跑阻塞的同步流式生成，逐段推给事件循环。

            走 SessionManager 的流式入口（而非直接调 agent），原因有三：
            1. 生成结束后要把**完整回复**写入 history / record，否则多轮上下文里
               模型看不到自己上一轮说了什么，归档复盘也拿不到任何 AI 产出；
            2. 首轮的外部资料准备（联网 / 知识库）与模型路由都在会话层；
            3. 工具调用事件由会话层统一收集，按序透出。
            `auto_finish=False` 保持「导出 / 定稿由用户显式触发」的既有交互不变。
            """
            try:
                stream = (
                    manager.start_stream(req.query, auto_finish=False)
                    if is_new
                    else manager.step_stream(req.query, auto_finish=False)
                )
                for piece in stream:
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

            def _forward_tool_event(event: dict) -> None:
                """把工具调用事件同时写入会话列表与 SSE 队列（实时可见）。"""
                record = {**event, "index": len(manager.tool_events)}
                manager.tool_events.append(record)
                asyncio.run_coroutine_threadsafe(queue.put({"__tool_event__": record}), loop).result()

            # 仅在工具调用路径下才会有事件；默认（开关关闭）下该监听器不会被触发。
            # 用户输入入栈、过程事件重置都由会话层在生成开始时统一处理，避免两处各写一遍。
            manager.agent.tool_event_listener = _forward_tool_event

            buffer: list[str] = []
            producer.start()
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    # 结构化输出（JD 匹配 / 面试复盘）在模型吐出第一个 token 前可能静默很久，
                    # 超过中间层（网关 OkHttp 读超时 / 反代）的等待上限就会被掐断连接。
                    # 发一条 SSE 注释作为心跳：不产生事件，前端解析时会自然忽略。
                    yield ": keep-alive\n\n"
                    continue
                if item is sentinel:
                    break
                if isinstance(item, dict) and "__tool_event__" in item:
                    # 工具调用过程实时透出：前端据此渲染可折叠时间线
                    payload = {"type": "tool_call", **item["__tool_event__"]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    continue
                if isinstance(item, Exception):
                    logger.error("流式生成失败：%s", item)
                    message = str(item)
                    buffer.append(message)
                    yield f"data: {json.dumps({'type': 'delta', 'text': message}, ensure_ascii=False)}\n\n"
                    break
                buffer.append(item)
                yield f"data: {json.dumps({'type': 'delta', 'text': item}, ensure_ascii=False)}\n\n"

            text = safe_output("".join(buffer))
            # 落库：归档接口（GET /sessions/{id} → jobseeker「归档到复盘」）读的是 DB 消息表。
            # 非流式接口一直有写，流式接口此前漏了这一步，导致复盘归档拿不到任何 AI 产出。
            # history / record 由会话层在生成结束时统一写入，这里不再重复追加。
            if text:
                db.add_message(sid, "assistant", text)
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
                # 兜底再给一次完整过程：前端即便漏收中间事件也能补齐时间线
                "tool_events": list(manager.tool_events),
                # 结构化产物（如 JD 匹配评分卡）随 done 下发，前端据字段渲染卡片而非纯文本
                "structured": _structured_payload(manager.agent),
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            # 还原监听器，避免后续非流式轮次往已关闭的事件循环投递事件
            manager.agent.tool_event_listener = manager.tool_events.append
            sem.release()

    return StreamingResponse(gen(), media_type="text/event-stream")


def _structured_payload(agent) -> dict | None:
    """取 Agent 的结构化产物并转成可 JSON 化的字典；没有则返回 None。

    只做透传，不在网关层理解业务字段：编排归 Agent，网关只负责搬运。
    """
    result = getattr(agent, "result", None)
    dump = getattr(result, "model_dump", None)
    if not callable(dump):
        return None
    try:
        return dump()
    except Exception as exc:  # noqa: BLE001
        logger.warning("结构化产物序列化失败，已跳过：%s", exc)
        return None


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
