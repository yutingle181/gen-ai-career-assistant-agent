"""Agent 基类：定义「单步会话」契约。

关键约定：Agent 内部**不允许出现 while 循环与 input()**，
多轮推进由 SessionManager 驱动，这样同一套逻辑既能跑 Web 也能跑 API。

工具调用的位置：Function Calling 是**单轮之内**的推理闭环，因此落在本类
（`respond` / `respond_stream` 内部分流），而不是交给 SessionManager——
后者负责的是「人机之间」的多轮推进，两者职责不重叠。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain_core.messages import BaseMessage, SystemMessage

from .. import config, metrics
from ..llm import llm_invoke
from ..logging_setup import get_logger
from ..prompts.personas import get_persona
from ..state import MODE_QA

logger = get_logger(__name__)


class BaseAgent:
    """所有场景 Agent 的基类。"""

    mode: str = MODE_QA
    artifact: str = "Q&A_Doubt_Session"
    one_shot: bool = False  # True 表示首轮即产出完整产物并自动结束
    needs_search: bool = False
    # True 表示该场景允许走 Function Calling 自主工具调用路径。
    # 仅声明意图：是否真正启用还取决于全局开关 config.ENABLE_TOOL_CALLING（默认关闭）。
    needs_tools: bool = False
    # True 表示该场景产物涉及外发 / 决策风险，需人工确认后才定稿（人机协同）。
    # 仅声明，不改生成逻辑；是否生效由 SessionManager 结合 config.ENABLE_HUMAN_CONFIRM 决定。
    requires_confirmation: bool = False

    def __init__(self, kb_name: str | None = None, retrieval_cfg=None) -> None:
        self.kb_name = kb_name
        self.retrieval_cfg = retrieval_cfg
        self.extra_context: str = ""
        self.citations: list[str] = []
        # 工具调用过程回调（由 SessionManager 注入，用于把过程透出到 SSE / 界面）
        self.tool_event_listener = None
        # 协作式取消信号（由 API 网关按需注入：用户点停止 / 客户端断开）。
        # 与 tool_event_listener 一样属于「按请求注入、用完恢复」的临时钩子，
        # 因此放在实例上而不是构造参数里——避免让所有调用方都多传一个参数。
        self.should_stop: Callable[[], bool] | None = None

    # ---------------------------------------------------------- 工具调用路径
    @property
    def use_tool_calling(self) -> bool:
        """显式检索路径 / 工具调用路径的切换开关。

        默认（开关关闭）完全走原有显式检索路径，行为与改造前一致；
        打开后由模型自主决定是否调用工具——两条路径构成 A/B 对比的实验组与对照组。
        """
        return bool(config.ENABLE_TOOL_CALLING and self.needs_tools)

    def build_tools(self, *, include_web: bool = True) -> list:
        """本场景可用的工具集（联网检索 + 知识库检索 + MCP）。

        `include_web=False` 供「只允许查内部资料」的场景使用（避免联网结果污染引用来源）。

        装配按**最小权限**：权限来自技能注册表（场景声明 ∩ 全局授予），
        缺权限的工具**不会挂上**，而不是挂上之后再拦；判定与计数见 `tools._allowed`。
        """
        if not self.needs_tools:
            return []
        from ..skills import effective_permissions
        from ..tools import get_agent_tools

        return get_agent_tools(
            self.kb_name,
            include_web=include_web,
            permissions=effective_permissions(self.mode),
        )

    # ---------------------------------------------------------- 人设
    @property
    def system_message(self) -> str:
        return get_persona(self.mode)

    # ---------------------------------------------------------- 上下文准备
    def prepare(self, query: str) -> str:
        """首轮可选的外部资料准备（联网检索 / 知识库检索），返回附加上下文。"""
        if not self.needs_search:
            return ""
        if self.use_tool_calling:
            # 联网改由模型自主决定，避免「前置检索一次 + 模型再检索一次」的重复开销
            logger.info("工具调用已启用，跳过前置联网检索 | %s", self.mode)
            return ""
        from ..skills import has_permission

        # 显式检索同样是"联网能力"，必须与工具装配走同一份授权判定，
        # 否则收紧权限只会关掉一半（工具不挂载，但前置检索照发）
        if not has_permission(self.mode, "net"):
            metrics.incr("skill.permission_denied")
            logger.warning("场景 %s 未获 net 权限，跳过前置联网检索", self.mode)
            return ""
        from ..tools import web_search

        results = web_search(query)
        if not results:
            return ""
        return f"\n\n【联网检索结果】（可能包含噪声，请甄别后使用）\n{results}"

    # ---------------------------------------------------------- 单轮生成
    def build_messages(self, history: Sequence[BaseMessage]) -> list[BaseMessage]:
        return [SystemMessage(content=self.system_message), *history]

    def respond(self, history: Sequence[BaseMessage], model: str | None = None,
                max_tokens: int | None = None, thread_id: str | None = None) -> str:
        """生成一轮回复；工具调用开关打开时走 ReAct 闭环。

        `thread_id` 透传给工具闭环，用于挂图级 checkpoint（同轮重试不重复执行工具）。
        """
        if self.use_tool_calling:
            from ..graph.tool_loop import run_with_tools

            return run_with_tools(
                self.build_messages(history),
                tools=self.build_tools(),
                model=model,
                max_tokens=max_tokens,
                on_event=self.tool_event_listener,
                thread_id=thread_id,
                should_stop=self.should_stop,
            )
        return llm_invoke(self.build_messages(history), model=model, max_tokens=max_tokens)

    def respond_stream(self, history: Sequence[BaseMessage], model: str | None = None,
                       max_tokens: int | None = None, thread_id: str | None = None):
        """流式生成一轮回复（逐段产出文本）。"""
        if self.use_tool_calling:
            from ..graph.tool_loop import stream_with_tools

            return stream_with_tools(
                self.build_messages(history),
                tools=self.build_tools(),
                model=model,
                max_tokens=max_tokens,
                on_event=self.tool_event_listener,
                thread_id=thread_id,
                should_stop=self.should_stop,
            )
        from ..llm import llm_stream

        return llm_stream(self.build_messages(history), model=model, max_tokens=max_tokens)

    # ---------------------------------------------------------- 收尾
    def finalize(self, transcript: str) -> str:
        """会话结束时追加到产物的内容（如面评、结构化职位表）。"""
        return ""

    @property
    def label(self) -> str:
        from ..state import MODE_LABELS

        return MODE_LABELS.get(self.mode, self.mode)
