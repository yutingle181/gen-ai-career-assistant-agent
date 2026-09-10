"""Agent 基类：定义「单步会话」契约。

关键约定：Agent 内部**不允许出现 while 循环与 input()**，
多轮推进由 SessionManager 驱动，这样同一套逻辑既能跑 Web 也能跑 API。
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, SystemMessage

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
    # True 表示该场景产物涉及外发 / 决策风险，需人工确认后才定稿（人机协同）。
    # 仅声明，不改生成逻辑；是否生效由 SessionManager 结合 config.ENABLE_HUMAN_CONFIRM 决定。
    requires_confirmation: bool = False

    def __init__(self, kb_name: str | None = None, retrieval_cfg=None) -> None:
        self.kb_name = kb_name
        self.retrieval_cfg = retrieval_cfg
        self.extra_context: str = ""
        self.citations: list[str] = []

    # ---------------------------------------------------------- 人设
    @property
    def system_message(self) -> str:
        return get_persona(self.mode)

    # ---------------------------------------------------------- 上下文准备
    def prepare(self, query: str) -> str:
        """首轮可选的外部资料准备（联网检索 / 知识库检索），返回附加上下文。"""
        if not self.needs_search:
            return ""
        from ..tools import web_search

        results = web_search(query)
        if not results:
            return ""
        return f"\n\n【联网检索结果】（可能包含噪声，请甄别后使用）\n{results}"

    # ---------------------------------------------------------- 单轮生成
    def build_messages(self, history: Sequence[BaseMessage]) -> list[BaseMessage]:
        return [SystemMessage(content=self.system_message), *history]

    def respond(self, history: Sequence[BaseMessage]) -> str:
        """生成一轮回复。"""
        return llm_invoke(self.build_messages(history))

    def respond_stream(self, history: Sequence[BaseMessage]):
        """流式生成一轮回复（逐段产出文本）。"""
        from ..llm import llm_stream

        return llm_stream(self.build_messages(history))

    # ---------------------------------------------------------- 收尾
    def finalize(self, transcript: str) -> str:
        """会话结束时追加到产物的内容（如面评、结构化职位表）。"""
        return ""

    @property
    def label(self) -> str:
        from ..state import MODE_LABELS

        return MODE_LABELS.get(self.mode, self.mode)
