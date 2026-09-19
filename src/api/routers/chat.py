"""对话路由：同步对话与 SSE 流式对话。"""

from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ... import config, metrics, tasks
from ...cache import Timer, get_cost_tracker
from ...graph.workflow import route_only
from ...logging_setup import get_logger
from ...safety import check_input, safe_output
from ...session import SessionManager
from ...state import MODE_LABELS
from .. import db
from ..deps import (
    build_retrieval_cfg,
    get_semaphore,
    get_session_manager,
    new_session_id,
    put_session,
    resolve_kb,
    restore_session,
)
from ..schemas import CancelRequest, ChatRequest, ChatResponse, FinishRequest

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)

# 静默心跳间隔（秒）：远小于网关 120s 读超时，保证长任务不会被中间层掐断
KEEPALIVE_SECONDS = 15.0


async def _acquire_slot(session_id: str) -> tasks.TaskHandle:
    """获取执行槽位（有界排队 + 排队超时），返回任务句柄。

    为什么不直接 `await sem.acquire()`：那样排队**无界且无反馈** ——
    客户端只看到「卡住」，服务端也无法拒绝，一个客户端并发打满队列就会拖垮所有人。
    这里给排队加上限（满 → 立刻 429）与上限等待时长（超时 → 503），
    并记录等待时长，让「排队慢」变成可观测的事实而不是用户的主观抱怨。
    """
    queued = tasks.counts()["queued"]
    if queued >= max(0, config.API_MAX_QUEUE):
        metrics.incr("api.queue.rejected")
        raise HTTPException(
            status_code=429,
            detail=f"服务繁忙（排队 {queued}/{config.API_MAX_QUEUE}），请稍后重试。",
            headers={"Retry-After": "3"},
        )

    handle = tasks.register(session_id)
    try:
        await asyncio.wait_for(
            get_semaphore().acquire(), timeout=max(1.0, config.API_QUEUE_TIMEOUT)
        )
    except asyncio.TimeoutError:
        metrics.incr("api.queue.timeout")
        handle.finish(tasks.FAILED)
        tasks.prune()
        raise HTTPException(
            status_code=503,
            detail=f"排队超过 {config.API_QUEUE_TIMEOUT:.0f}s 仍未获得执行槽位，请稍后重试。",
            headers={"Retry-After": "5"},
        ) from None
    wait_ms = handle.mark_running()
    metrics.observe("api.queue.wait_ms", wait_ms)
    if wait_ms >= 1000:
        logger.info("排队等待 %dms 后开始执行 | session=%s", wait_ms, session_id)
    return handle


def _release_slot(handle: tasks.TaskHandle, *, disconnected: bool = False) -> None:
    """释放槽位并落终态（任何路径都必须走到，否则并发额度会被漏掉）。

    `disconnected=True` 用于「生成器被关闭」这条路径（客户端断开 / 关页面）：
    此时任务**还在跑**，所以必须先发取消信号，producer 线程才会在下一个检查点停下；
    否则用户走了、额度还在烧。正常跑完的任务已落终态，这里自然成为空操作。
    """
    if disconnected:
        tasks.cancel_if_running(handle.session_id, "client_gone")
    if handle.cancelled:
        metrics.incr("task.cancelled")
    handle.finish(tasks.CANCELLED if handle.cancelled else tasks.DONE)
    tasks.prune()
    get_semaphore().release()


def _sse(payload: dict) -> str:
    """拼一帧 SSE（统一格式，避免各处重复 json.dumps + 空行约定）。"""
    import json

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _close_stream(stream) -> None:  # noqa: ANN001
    """关闭流式生成器：把 GeneratorExit 传进会话层与底层模型流，真正停下 token 消耗。

    只「不再往外推」是不够的 —— 后台会继续把这一轮生成完，用户的额度照样被扣。
    """
    close = getattr(stream, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("关闭流式生成器失败（忽略）：%s", exc)


def _resolve_mode(req: ChatRequest) -> str:
    if req.mode:
        return req.mode
    return route_only(req.query).get("mode", "qa")


def _ensure_session(req: ChatRequest) -> tuple[str, SessionManager, bool]:
    """获取 / 恢复 / 新建会话，返回 (session_id, manager, is_new)。"""
    if req.session_id:
        manager = get_session_manager(req.session_id)
        if manager is not None:
            return req.session_id, manager, False
        # 进程内没有该会话：先尝试从持久化恢复（进程重启、多 worker 场景）
        manager = restore_session(req.session_id)
        if manager is not None:
            # 岗位 / 简历这类 user_context 每轮都可能更新，恢复后按本次请求覆盖
            if req.user_context:
                manager.user_context = req.user_context
                manager.agent.extra_context = req.user_context
            return req.session_id, manager, False

    mode = _resolve_mode(req)
    kb_name = resolve_kb(req.kb_name) if mode == "knowledge" else req.kb_name
    cfg = build_retrieval_cfg(req.top_k, req.reranker, req.use_bm25, req.use_vector)
    manager = SessionManager(mode, kb_name=kb_name, retrieval_cfg=cfg, user_context=req.user_context)
    sid = req.session_id or new_session_id()
    manager.session_id = sid  # 供产物幂等命名 + 单轮工具闭环的 checkpoint thread 键
    if db.get_session(sid) is None:
        db.create_session(sid, mode, kb_name or "")
    put_session(sid, manager)
    logger.info("新建会话 | id=%s mode=%s", sid, mode)
    return sid, manager, True


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """一轮对话（非流式）。

    取消语义（与非流式接口的能力边界有关，如实说明）：非流式只有**开始前**的检查点，
    一旦进入模型调用就不会被中断 —— 需要中途取消请用 `/chat/stream` + `/chat/cancel`。
    """
    ok, reason = check_input(req.query)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    sid, manager, is_new = _ensure_session(req)
    handle = await _acquire_slot(sid)
    try:
        if handle.cancelled:
            # 排队期间用户已取消：直接返回，不该再花一次模型调用
            return _cancelled_response(sid, manager)

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
    finally:
        _release_slot(handle)


def _cancelled_response(sid: str, manager: SessionManager) -> ChatResponse:
    """取消后的统一回执（非流式路径用）。"""
    return ChatResponse(
        session_id=sid,
        mode=manager.mode,
        mode_label=MODE_LABELS.get(manager.mode, manager.mode),
        message="（本轮已取消）",
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式对话：先推送 mode 事件，再逐段推送文本。"""
    import json

    ok, reason = check_input(req.query)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    sid, manager, is_new = _ensure_session(req)
    handle = await _acquire_slot(sid)

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

            逐段检查取消信号：用户点了停止 / 关了页面时，**关闭生成器**而不只是不再往外推——
            关闭会把 GeneratorExit 传进 `manager.*_stream` 与底层的模型流式响应，
            真正的 token 消耗才会停下来（否则只是「前端不显示了，后台照烧」）。
            """
            try:
                stream = (
                    manager.start_stream(req.query, auto_finish=False)
                    if is_new
                    else manager.step_stream(req.query, auto_finish=False)
                )
                for piece in stream:
                    if handle.should_stop():
                        _close_stream(stream)
                        break
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

            if handle.cancelled:
                # 排队期间就被取消：不写用户消息、不进模型，直接把结论告诉客户端
                yield _sse({"type": "cancelled", "session_id": sid, "reason": handle.cancel_reason})
                return

            db.add_message(sid, "user", req.query)

            def _forward_tool_event(event: dict) -> None:
                """把工具调用事件同时写入会话列表与 SSE 队列（实时可见）。"""
                record = {**event, "index": len(manager.tool_events)}
                manager.tool_events.append(record)
                asyncio.run_coroutine_threadsafe(queue.put({"__tool_event__": record}), loop).result()

            # 仅在工具调用路径下才会有事件；默认（开关关闭）下该监听器不会被触发。
            # 用户输入入栈、过程事件重置都由会话层在生成开始时统一处理，避免两处各写一遍。
            manager.agent.tool_event_listener = _forward_tool_event
            # 取消信号一并注入：图执行层在「工具轮次之间」检查它，
            # 才能做到「用户点了停止，模型不再继续调工具」而不只是停掉前端渲染。
            manager.agent.should_stop = handle.should_stop

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
            if handle.cancelled:
                # 取消也要留痕：否则用户回到会话里会以为「这一轮什么都没发生」。
                # 但只写 DB（归档可见），不进模型 history —— 半截回复进上下文会污染下一轮。
                text = f"{text}\n\n（本轮已取消，以上为已生成的部分内容）".strip()
            # 落库：归档接口（GET /sessions/{id} → jobseeker「归档到复盘」）读的是 DB 消息表。
            # 非流式接口一直有写，流式接口此前漏了这一步，导致复盘归档拿不到任何 AI 产出。
            # history / record 由会话层在生成结束时统一写入，这里不再重复追加。
            if text:
                db.add_message(sid, "assistant", text)
            if manager.agent.one_shot and not handle.cancelled:
                # 取消时不自动收尾：别用半截内容生成定稿产物
                manager.finish()

            if handle.cancelled:
                yield _sse({"type": "cancelled", "session_id": sid, "reason": handle.cancel_reason})

            payload = {
                "type": "done",
                "cancelled": handle.cancelled,
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
            # 正常跑完：先落终态，finally 里的「断连兜底」才不会把这一轮误算成取消
            handle.finish(tasks.CANCELLED if handle.cancelled else tasks.DONE)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            # 还原监听器，避免后续非流式轮次往已关闭的事件循环投递事件
            manager.agent.tool_event_listener = manager.tool_events.append
            manager.agent.should_stop = None
            # 生成器被关闭（客户端断开）时任务还在跑 → 由这里发出取消信号；
            # 正常跑完的任务已落终态，此处是空操作。
            _release_slot(handle, disconnected=not handle.finished)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/cancel")
async def chat_cancel(req: CancelRequest) -> dict:
    """取消一轮在途对话（用户点「停止」/ 客户端主动放弃这一轮）。

    返回值语义要看清：`cancelled=true` 只代表**取消信号已发出**，
    线程会在下一个检查点才真正停下（流式分片之间 / 工具轮次之间），
    所以调用方不要据此立刻假设资源已释放、额度已停止消耗。
    """
    ok = tasks.cancel(req.session_id, reason="user")
    handle = tasks.current(req.session_id)
    return {
        "cancelled": ok,
        "status": handle.status if handle is not None else "not_found",
        "detail": (
            "取消信号已发出，将在下一个检查点停止"
            if ok
            else "该会话没有在途任务（可能已结束或不存在）"
        ),
    }


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
