"""知识库问答 Agent：基于 RAG 引擎回答，输出带引用来源。"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..llm import llm_invoke
from ..logging_setup import get_logger
from ..rag.pipeline import RAGPipeline, RetrievalConfig
from ..rag.registry import get_registry
from ..state import MODE_KNOWLEDGE
from .base import BaseAgent

logger = get_logger(__name__)


class KnowledgeAgent(BaseAgent):
    """知识库问答：每一轮都走「检索 -> 重排 -> 带引用生成」。"""

    mode = MODE_KNOWLEDGE
    artifact = "Knowledge_QA"
    one_shot = False
    needs_search = False
    # 允许 Function Calling 路径：是否检索、检索几次由模型自主决定。
    # 开关（config.ENABLE_TOOL_CALLING）关闭时完全走原有固定检索链路。
    needs_tools = True

    def __init__(self, kb_name: str | None = None, retrieval_cfg: RetrievalConfig | None = None):
        super().__init__(kb_name=kb_name, retrieval_cfg=retrieval_cfg)
        self.last_chunks = []

    @property
    def pipeline(self) -> RAGPipeline | None:
        return get_registry().get(self.kb_name) if self.kb_name else None

    def build_tools(self) -> list:
        """知识库场景只暴露知识库检索工具：联网结果会污染引用来源。"""
        if not self.needs_tools or not self.kb_name:
            return []
        from ..tools import get_agent_tools

        return get_agent_tools(self.kb_name, include_web=False)

    def prepare(self, query: str) -> str:
        """检查知识库是否已建库，给出明确中文提示。"""
        if not self.kb_name:
            return "\n\n【提示】尚未选择知识库。"
        pipe = self.pipeline
        if pipe is None or not pipe.ready:
            return "\n\n【提示】当前知识库尚未建库，请先上传文档。"
        return ""

    def respond(self, history: Sequence[BaseMessage], model: str | None = None,
                max_tokens: int | None = None) -> str:
        """优先走 RAG；知识库不可用时回退为普通问答。"""
        if self.use_tool_calling:
            # 工具调用路径：由模型自主决定何时检索知识库，检索结果自带来源标注
            return super().respond(history, model=model, max_tokens=max_tokens)

        query = self._last_user_text(history)
        pipe = self.pipeline

        if pipe is None or not pipe.ready:
            return (
                "当前知识库不可用（未选择或尚未建库）。"
                "请先在左侧「知识库」区域上传文档并完成建库，再向我提问。\n\n"
                "在你建库之前，我也可以先按通用知识回答你的问题。"
            )

        cfg = self.retrieval_cfg or RetrievalConfig()
        try:
            result = pipe.answer(query, cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("知识库问答失败，回退为普通问答：%s", exc)
            return f"知识库检索失败（{type(exc).__name__}），我已回退为通用问答：\n\n" + llm_invoke(
                [SystemMessage(content=self.system_message), *history],
                model=model,
                max_tokens=max_tokens,
            )

        self.last_chunks = result.chunks
        self.citations = [c.source for c in result.citations]

        citations_text = ""
        if result.citations:
            citations_text = "\n\n---\n\n**引用来源**\n" + "\n".join(
                f"- [{c.index}] {c.source}" + (f" 第{c.page}页" if c.page else "")
                for c in result.citations
            )
        return result.answer + citations_text

    def respond_stream(self, history: Sequence[BaseMessage], model: str | None = None,
                       max_tokens: int | None = None):
        """知识库问答为一次性生成，流式退化为分段推送。"""
        text = self.respond(history, model=model, max_tokens=max_tokens)
        step = 40
        for i in range(0, len(text), step):
            yield text[i : i + step]

    @staticmethod
    def _last_user_text(history: Sequence[BaseMessage]) -> str:
        for msg in reversed(list(history)):
            if isinstance(msg, HumanMessage):
                return msg.content or ""
            if getattr(msg, "type", "") == "human":
                return getattr(msg, "content", "") or ""
        return ""
