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
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .. import config, metrics
from ..llm import get_chat_model, llm_invoke
from ..logging_setup import get_logger
from ..telemetry import span
from ..tools import degraded_kind
from .checkpoint import get_checkpointer

logger = get_logger(__name__)

# 工具调用事件回调：payload 含 name / ok / elapsed_ms / args_summary / result_summary
ToolEventListener = Callable[[dict[str, Any]], None]

_SUMMARY_LIMIT = 120

# 收束指令：图内 finalize 节点与图外兜底共用同一句，保证「到上限时的回答要求」只有一处定义。
_FINALIZE_NOTE = "请立即基于已有信息给出最终回答，不要再调用任何工具。"

# 图级递归上限触发、且模型仍拿不出文本时的兜底文案。
# 取舍：宁可给一句「明确的阶段性结论 + 怎么继续」，也不把 GraphRecursionError 抛给用户——
# 对使用者来说「被硬性截断」和「服务崩了」是完全不同的体验。
RECURSION_FALLBACK_TEXT = (
    "（本轮工具调用已达执行上限，我先基于现有信息给出结论；"
    "如果还需要更细的结果，请把问题拆得更具体一些再发一次。）"
)


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


def _forced_finalize(model, messages: Sequence[BaseMessage]) -> str:
    """图外强制收束：摘掉工具定义再问一次，拿不到文本就给兜底文案。

    与图内 `finalize` 节点同源（共用 `_FINALIZE_NOTE`），区别是它不经过图执行，
    因此能用于「图级递归上限」这类图外终止路径——这也是本函数存在的唯一理由。
    """
    try:
        resp = model.invoke([*messages, HumanMessage(content=_FINALIZE_NOTE)])
        text = _message_text(resp).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("图外强制收束失败，改用兜底文案：%s", exc)
        text = ""
    return text or RECURSION_FALLBACK_TEXT


def _invoke_tool_graph(app, initial: dict, graph_config: dict, resumable: bool) -> dict:
    """执行工具闭环；同一 thread 已有 checkpoint 时改为「续跑」。

    为什么必须单独处理：挂了 checkpointer 之后，输入会被**合并**进已有状态，
    把同一批消息再喂一遍会让历史翻倍。所以「重试同一轮」时只能传 None（纯续跑），
    而不是重传消息。三种情形：

    - 该 thread 尚无 checkpoint → 正常启动；
    - 有 checkpoint 且还有待执行节点 → 传 None 续跑未完成的那一轮；
    - 有 checkpoint 且该轮已跑完 → 直接复用结果，**不重复执行模型与工具**
      （这正是「重试不产生重复副作用」的落点）。
    """
    if not resumable:
        return app.invoke(initial, config=graph_config)

    thread_id = (graph_config.get("configurable") or {}).get("thread_id", "")
    snapshot = app.get_state(graph_config)
    if snapshot and snapshot.values.get("messages"):
        if snapshot.next:
            logger.info("检测到未完成的 checkpoint，续跑本轮工具闭环 | thread=%s", thread_id)
            return app.invoke(None, config=graph_config)
        logger.info("本轮已有完整 checkpoint，直接复用结果（不重复执行工具）| thread=%s", thread_id)
        return dict(snapshot.values)
    return app.invoke(initial, config=graph_config)


def run_with_tools(
    messages: Sequence[BaseMessage],
    *,
    tools: list,
    model: str | None = None,
    max_tokens: int | None = None,
    max_steps: int | None = None,
    on_event: ToolEventListener | None = None,
    thread_id: str | None = None,
) -> str:
    """在轮次上限内完成「模型 → 工具 → 模型」闭环，返回最终自然语言回答。

    无工具或模型不支持 tool calling 时，自动退回一次普通调用（不抛错），
    保证「打开开关但环境不具备」的场景仍能正常作答。

    `thread_id`（通常是「会话 id + 轮次」）用于挂图级 checkpoint：
    同一 thread 再次调用会复用/续跑已有状态，而不是重新执行工具。
    不传则完全不启用 checkpoint，行为与改动前一致。
    """
    if not tools:
        return llm_invoke(messages, model=model, max_tokens=max_tokens)

    # 指标口径：agent.run 是「跑过一次工具闭环」的样本数，
    # 超轮次率与递归终止率都以它为分母（见 src/metrics.py 的 _RATE_RULES）。
    metrics.incr("agent.run")
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
        metrics.incr("agent.tool_round")
        started = time.monotonic()
        out = tool_node.invoke({"messages": [last]})
        elapsed_ms = int((time.monotonic() - started) * 1000)
        produced = list(out.get("messages", []))
        for idx, call in enumerate(calls):
            result_msg = produced[idx] if idx < len(produced) else None
            text = _message_text(result_msg)
            # 工具「软失败」也要如实标记：降级文本带统一标记（【工具降级:xxx】），
            # 事件里置 ok=False 并附失败分类，前端时间线与指标才能区分「查到了」和「没查到」。
            kind = degraded_kind(text)
            ok = getattr(result_msg, "status", "success") != "error" and not kind
            name = str(call.get("name") or "unknown")
            with span(
                "agent.tool_call",
                {
                    "tool.name": name,
                    "tool.step": step,
                    "tool.ok": ok,
                    "tool.error_kind": kind,
                    "tool.elapsed_ms": elapsed_ms,
                },
            ):
                _emit(
                    on_event,
                    {
                        "name": name,
                        "ok": ok,
                        "error_kind": kind,
                        "elapsed_ms": elapsed_ms,
                        "args_summary": _clip(json.dumps(call.get("args") or {}, ensure_ascii=False)),
                        "result_summary": _clip(text),
                    },
                )
        logger.info("工具调用完成 | 第 %d 轮 | %d 个工具 | %dms", step, len(calls), elapsed_ms)
        return {"messages": produced}

    def finalize(state: ToolLoopState) -> dict:
        """超限收束：摘掉工具定义再问一次，强制给出自然语言结论。"""
        note = HumanMessage(content=_FINALIZE_NOTE)
        with span("agent.tool_call.finalize", {"tool.step": int(state.get("steps", 0))}):
            resp = plain.invoke([*state["messages"], note])
        return {"messages": [resp]}

    def route(state: ToolLoopState) -> str:
        last = state["messages"][-1]
        if not (getattr(last, "tool_calls", None) or []):
            return END
        if int(state.get("steps", 0)) >= limit:
            logger.warning("工具调用达到上限 %d 轮，强制收束为直接作答", limit)
            metrics.incr("agent.step_limit")
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

    # 图级 checkpoint（P1-4）：只在调用方给了 thread_id 时挂载——
    # 没有会话标识就没有隔离键，挂上去只会让不同会话互相污染状态。
    saver = get_checkpointer()[0] if thread_id else None
    app = graph.compile(checkpointer=saver) if saver is not None else graph.compile()
    # 第二层兜底（图级）：即使条件边没能按预期收束，LangGraph 也会在 recursion_limit 处硬停。
    # 这里捕获该异常并改走一次「图外强制收束」，保证用户始终拿到可读结论而不是栈。
    graph_config: dict[str, Any] = {"recursion_limit": max(1, config.TOOL_RECURSION_LIMIT)}
    if saver is not None:
        graph_config["configurable"] = {"thread_id": thread_id}
    try:
        final_state = _invoke_tool_graph(
            app,
            {"messages": list(messages), "steps": 0},
            graph_config,
            resumable=saver is not None,
        )
    except GraphRecursionError:
        logger.warning(
            "图级递归达到上限 %d（单轮轮次上限 %d），改为强制收束作答",
            graph_config["recursion_limit"],
            limit,
        )
        metrics.incr("agent.recursion_stop")
        with span(
            "agent.tool_call.recursion_stop",
            {
                "tool.max_steps": limit,
                "graph.recursion_limit": graph_config["recursion_limit"],
            },
        ):
            return _forced_finalize(plain, list(messages))
    metrics.incr("agent.ok")
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
    thread_id: str | None = None,
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
        thread_id=thread_id,
    )
    for start in range(0, len(text), max(1, chunk_size)):
        yield text[start : start + max(1, chunk_size)]
