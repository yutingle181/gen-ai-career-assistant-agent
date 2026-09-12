"""工具调用子图：单轮之内的 ReAct 闭环（模型 → 工具 → 模型 … → 最终作答）。

为什么是 `StateGraph` 而不是 `AgentExecutor` 或手写 `while`：
`base.py` 明确约定「Agent 内部不允许出现 while 循环」，目的是把**多轮人机推进**
的职责收敛给 `SessionManager`，从而复用同一套逻辑跑 Web 与 API。工具调用是
**单轮之内**的推理闭环，与「多轮人机推进」是两件事——用条件边表达 ReAct 既满足
语义，又不破坏既有约定，也避免引入与项目风格不一致的额外抽象层。

三条安全边界（保证工具调用不会把体验拖垮）：
1. 轮次上限 `TOOL_CALLING_MAX_STEPS`：达到即强制收束为自然语言作答；
2. 单步工具执行沿用工具层已内建的守护线程超时（`WEB_SEARCH_TIMEOUT`）；
3. 工具异常包装为一句简短中文回灌，不让堆栈进入上下文、推高 token。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .. import config
from ..llm import get_chat_model, llm_invoke
from ..logging_setup import get_logger
from ..telemetry import span

logger = get_logger(__name__)

# 工具调用事件回调：payload 含 name / ok / elapsed_ms / args_summary / result_summary
ToolEventListener = Callable[[dict[str, Any]], None]

_SUMMARY_LIMIT = 120


class ToolLoopState(TypedDict, total=False):
    """子图状态：消息序列（追加式）+ 已消耗的模型轮次。"""

    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


def _message_text(msg: BaseMessage | None) -> str:
    """取出消息文本，兼容部分兼容接口返回 list 形态的 content。"""
    if msg is None:
        return ""
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


def _clip(text: str, limit: int = _SUMMARY_LIMIT) -> str:
    """压缩成单行摘要——事件与日志只带摘要，不落查询原文与工具全文。"""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _tool_error_message(exc: Exception) -> str:
    """工具异常回灌：只给一句可操作的中文，堆栈留在日志里。"""
    logger.warning("工具执行失败 | %s | %s", type(exc).__name__, exc)
    return f"工具执行失败（{type(exc).__name__}），请基于已有信息作答，或换一种方式获取信息。"


def _emit(on_event: ToolEventListener | None, payload: dict[str, Any]) -> None:
    """上报工具调用事件；回调异常绝不影响主流程。"""
    if on_event is None:
        return
    try:
        on_event(payload)
    except Exception as exc:  # noqa: BLE001
        logger.debug("工具调用事件回调异常已忽略：%s", exc)


def _final_text(messages: Sequence[BaseMessage]) -> str:
    """自后向前取第一条非空 AIMessage 文本（收束轮的结果优先）。"""
    for msg in reversed(list(messages)):
        if isinstance(msg, AIMessage):
            text = _message_text(msg).strip()
            if text:
                return text
    return ""


def run_with_tools(
    messages: Sequence[BaseMessage],
    *,
    tools: list,
    model: str | None = None,
    max_tokens: int | None = None,
    max_steps: int | None = None,
    on_event: ToolEventListener | None = None,
) -> str:
    """在轮次上限内完成「模型 → 工具 → 模型」闭环，返回最终自然语言回答。

    无工具或模型不支持 tool calling 时，自动退回一次普通调用（不抛错），
    保证「打开开关但环境不具备」的场景仍能正常作答。
    """
    if not tools:
        return llm_invoke(messages, model=model, max_tokens=max_tokens)

    limit = max(1, config.TOOL_CALLING_MAX_STEPS if max_steps is None else max_steps)
    plain = get_chat_model(model=model, max_tokens=max_tokens)
    try:
        bound = plain.bind_tools(tools)
    except Exception as exc:  # noqa: BLE001
        logger.warning("模型不支持工具调用，已退回普通作答：%s", exc)
        return llm_invoke(messages, model=model, max_tokens=max_tokens)

    tool_node = ToolNode(tools, handle_tool_errors=_tool_error_message)

    def assistant(state: ToolLoopState) -> dict:
        """模型节点：一次推理，可能产出 tool_calls，也可能是最终回答。"""
        step = int(state.get("steps", 0)) + 1
        with span("agent.tool_call.round", {"tool.step": step, "tool.max_steps": limit}):
            resp = bound.invoke(list(state["messages"]))
        return {"messages": [resp], "steps": step}

    def tools_step(state: ToolLoopState) -> dict:
        """工具节点：执行本轮全部 tool_calls，并把过程透出为事件。"""
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        step = int(state.get("steps", 0))
        started = time.monotonic()
        out = tool_node.invoke({"messages": [last]})
        elapsed_ms = int((time.monotonic() - started) * 1000)
        produced = list(out.get("messages", []))
        for idx, call in enumerate(calls):
            result_msg = produced[idx] if idx < len(produced) else None
            ok = getattr(result_msg, "status", "success") != "error"
            name = str(call.get("name") or "unknown")
            with span(
                "agent.tool_call",
                {"tool.name": name, "tool.step": step, "tool.ok": ok, "tool.elapsed_ms": elapsed_ms},
            ):
                _emit(
                    on_event,
                    {
                        "name": name,
                        "ok": ok,
                        "elapsed_ms": elapsed_ms,
                        "args_summary": _clip(json.dumps(call.get("args") or {}, ensure_ascii=False)),
                        "result_summary": _clip(_message_text(result_msg)),
                    },
                )
        logger.info("工具调用完成 | 第 %d 轮 | %d 个工具 | %dms", step, len(calls), elapsed_ms)
        return {"messages": produced}

    def finalize(state: ToolLoopState) -> dict:
        """超限收束：摘掉工具定义再问一次，强制给出自然语言结论。"""
        note = HumanMessage(content="请立即基于已有信息给出最终回答，不要再调用任何工具。")
        with span("agent.tool_call.finalize", {"tool.step": int(state.get("steps", 0))}):
            resp = plain.invoke([*state["messages"], note])
        return {"messages": [resp]}

    def route(state: ToolLoopState) -> str:
        last = state["messages"][-1]
        if not (getattr(last, "tool_calls", None) or []):
            return END
        if int(state.get("steps", 0)) >= limit:
            logger.warning("工具调用达到上限 %d 轮，强制收束为直接作答", limit)
            return "finalize"
        return "tools"

    graph = StateGraph(ToolLoopState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", tools_step)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(
        "assistant", route, {"tools": "tools", "finalize": "finalize", END: END}
    )
    graph.add_edge("tools", "assistant")
    graph.add_edge("finalize", END)

    app = graph.compile()
    final_state = app.invoke({"messages": list(messages), "steps": 0})
    return _final_text(final_state["messages"])


def stream_with_tools(
    messages: Sequence[BaseMessage],
    *,
    tools: list,
    model: str | None = None,
    max_tokens: int | None = None,
    max_steps: int | None = None,
    on_event: ToolEventListener | None = None,
    chunk_size: int = 24,
):
    """工具调用路径的流式外壳：先跑完闭环，再分片吐出最终答案。

    取舍说明：工具调用链路本身是「决策轮 + 执行轮」的多次模型往返，逐 token
    流式收益有限，反而会让「哪些分片属于过程说明、哪些属于最终答案」变得不可控。
    因此这里诚实地选择「闭环后分片输出」——用户看到的是稳定的渐显效果，
    真正的过程可见性由 `on_event` 透出的工具调用时间线承担（前端与 Streamlit 展示）。
    """
    text = run_with_tools(
        messages,
        tools=tools,
        model=model,
        max_tokens=max_tokens,
        max_steps=max_steps,
        on_event=on_event,
    )
    for start in range(0, len(text), max(1, chunk_size)):
        yield text[start : start + max(1, chunk_size)]
