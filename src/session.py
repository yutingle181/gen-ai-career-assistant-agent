"""会话管理：替代原 Notebook 中 `while True + input()` 的阻塞循环。

驱动模型：UI / API 每次收到一条用户输入，就调用 `step()` 推进一步。
Agent 内部不再持有循环，因此同一套逻辑可同时服务 Streamlit 与 FastAPI。
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from . import config
from .agents import create_agent
from .logging_setup import get_logger, summarize
from .rag.pipeline import RetrievalConfig
from .safety import check_high_risk, check_input, safe_output
from .state import MODE_LABELS
from .storage import DRAFT_SUFFIX, save_file
from .telemetry import span

logger = get_logger(__name__)


class SessionManager:
    """一次完整会话（可能是单轮生成，也可能是多轮对话）。"""

    def __init__(
        self,
        mode: str,
        kb_name: str | None = None,
        retrieval_cfg: RetrievalConfig | None = None,
        user_context: str | None = None,
    ) -> None:
        self.mode = mode
        self.kb_name = kb_name
        self.retrieval_cfg = retrieval_cfg
        # 用户档案（岗位 JD + 关联简历），由后端注入；为空时行为与之前完全一致
        self.user_context = user_context or ""
        self.agent = create_agent(mode, kb_name=kb_name, retrieval_cfg=retrieval_cfg)
        self.history: list = []
        self.record: list[str] = []
        self.started = False
        self.finished = False
        self.artifact_path: str | None = None
        self.error: str = ""
        # ---- 人机协同：草稿态 ----
        self.awaiting_confirmation = False  # 草稿已产出、等待人工确认
        self.draft_path: str | None = None  # 草稿产物路径
        self.high_risk_topic: str = ""  # 命中的高风险话题（空 = 未命中）
        self._confirmed = False  # 是否已人工确认过

    # ------------------------------------------------------------ 生命周期
    def start(self, user_query: str) -> str:
        """首轮：准备上下文 -> 生成 -> 一次性模式自动收尾。"""
        ok, reason = check_input(user_query)
        if not ok:
            self.error = reason
            return f"（{reason}）"

        self._detect_high_risk(user_query)

        self.record.append(f"**用户**：{user_query}\n")
        context = self.agent.prepare(user_query)
        self.history.append(HumanMessage(content=f"{user_query}{context}"))
        self.started = True

        reply = self._generate()
        if self.agent.one_shot:
            self.finish()
        return reply

    def step(self, user_text: str) -> str:
        """后续轮：追加用户输入 -> 生成。"""
        if self.finished:
            return "本轮会话已结束并导出。如需继续，请点击「新建会话」。"
        if not self.started:
            return self.start(user_text)

        ok, reason = check_input(user_text)
        if not ok:
            return f"（{reason}）"

        self._detect_high_risk(user_text)

        self.record.append(f"\n**用户**：{user_text}\n")
        self.history.append(HumanMessage(content=user_text))
        return self._generate()

    def start_stream(self, user_query: str):
        """首轮（流式版）：与 start() 流程一致，只是逐段产出文本。"""
        ok, reason = check_input(user_query)
        if not ok:
            self.error = reason
            yield f"（{reason}）"
            return

        self._detect_high_risk(user_query)

        self.record.append(f"**用户**：{user_query}\n")
        context = self.agent.prepare(user_query)
        self.history.append(HumanMessage(content=f"{user_query}{context}"))
        self.started = True

        yield from self._generate_stream()
        if self.agent.one_shot:
            self.finish()

    def step_stream(self, user_text: str):
        """后续轮（流式版）：与 step() 流程一致，只是逐段产出文本。"""
        if self.finished:
            yield "本轮会话已结束并导出。如需继续，请点击「新建会话」。"
            return
        if not self.started:
            yield from self.start_stream(user_text)
            return

        ok, reason = check_input(user_text)
        if not ok:
            yield f"（{reason}）"
            return

        self._detect_high_risk(user_text)

        self.record.append(f"\n**用户**：{user_text}\n")
        self.history.append(HumanMessage(content=user_text))
        yield from self._generate_stream()

    @property
    def requires_confirmation(self) -> bool:
        """本会话产物是否需要人工确认后才定稿。

        命中条件：总开关打开 且（场景本身高风险 或 用户输入命中高风险话题）。
        已确认过则不再需要。
        """
        if not config.ENABLE_HUMAN_CONFIRM:
            return False
        if self._confirmed:
            return False
        return bool(getattr(self.agent, "requires_confirmation", False)) or bool(self.high_risk_topic)

    def _detect_high_risk(self, text: str) -> None:
        """记录用户输入命中的高风险话题（只打标，不拦截，内容照常生成）。"""
        hit, topic = check_high_risk(text)
        if hit and not self.high_risk_topic:
            self.high_risk_topic = topic

    def _build_content(self) -> str:
        """拼装产物正文，草稿与定稿共用（保证两者内容一致）。"""
        transcript = "\n".join(self.record).strip()
        extra = ""
        try:
            extra = self.agent.finalize(transcript) or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("收尾生成失败，跳过：%s", exc)

        label = MODE_LABELS.get(self.mode, self.mode)
        header = (
            f"# {label} 会话记录\n\n"
            f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- 会话模式：{label}（{self.mode}）\n"
        )
        if self.kb_name:
            header += f"- 知识库：{self.kb_name}\n"
        if self.retrieval_cfg:
            header += (
                f"- 检索配置：BM25={self.retrieval_cfg.use_bm25} "
                f"向量={self.retrieval_cfg.use_vector} "
                f"融合={self.retrieval_cfg.fusion} "
                f"重排={self.retrieval_cfg.reranker} top_k={self.retrieval_cfg.top_k}\n"
            )

        return f"{header}\n---\n\n{transcript}\n{extra}"

    def finish(self) -> str | None:
        """结束会话。

        - 需人工确认的场景：先落**草稿**并置 awaiting_confirmation，返回草稿路径；
        - 其余场景：直接落定稿并置 finished。

        两条路径都幂等：重复调用不会重复落盘（草稿态下返回已有草稿）。
        """
        if self.finished and self.artifact_path:
            return self.artifact_path
        if self.awaiting_confirmation and self.draft_path:
            return self.draft_path  # 已在待确认态，不重复落草稿

        content = self._build_content()
        if self.requires_confirmation:
            self.draft_path = save_file(content, f"{self.agent.artifact}{DRAFT_SUFFIX}")
            self.awaiting_confirmation = True
            logger.info("草稿已产出，等待人工确认 | mode=%s | %s", self.mode, self.draft_path)
            return self.draft_path

        self.artifact_path = save_file(content, self.agent.artifact)
        self.finished = True
        logger.info("会话已导出 | mode=%s | %s", self.mode, self.artifact_path)
        return self.artifact_path

    def confirm(self) -> str | None:
        """人工确认：把草稿定稿为最终产物。

        草稿与定稿存为两个文件，保留「审阅前 / 审阅后」的可追溯证据。
        """
        if self.finished and self.artifact_path:
            return self.artifact_path

        self._confirmed = True
        self.artifact_path = save_file(self._build_content(), self.agent.artifact)
        self.finished = True
        self.awaiting_confirmation = False
        logger.info("已人工确认定稿 | mode=%s | %s", self.mode, self.artifact_path)
        return self.artifact_path

    # ------------------------------------------------------------ 内部
    def _generate(self) -> str:
        with span(
            "agent.respond",
            {
                "agent.mode": self.mode,
                "agent.kb": self.kb_name or "",
                "agent.multi_turn": not self.agent.one_shot,
            },
        ) as sp:
            trimmed = self._trim_history()
            try:
                reply = self.agent.respond(trimmed)
            except Exception as exc:  # noqa: BLE001
                sp.set_attribute("error", True)
                sp.set_attribute("error.type", type(exc).__name__)
                logger.error("生成失败：%s", exc)
                self.error = f"{type(exc).__name__}: {exc}"
                hint = "未配置 OPENAI_API_KEY" if "OPENAI_API_KEY" in str(exc) else "模型调用失败"
                reply = f"（{hint}，请检查 .env 配置后重试。错误详情：{type(exc).__name__}）"

            reply = safe_output(reply or "")
            sp.set_attribute("agent.reply_chars", len(reply or ""))
            self.history.append(AIMessage(content=reply))
            self.record.append(f"\n**助手**：{reply}\n")
            return reply

    def _generate_stream(self):
        """流式生成：先逐段产出，生成结束后才把完整回复入栈。

        注意：历史与会话记录必须在生成结束后再写入，
        否则多轮对话会把「半截回复」当作下一轮的上下文。
        """
        with span(
            "agent.respond",
            {
                "agent.mode": self.mode,
                "agent.kb": self.kb_name or "",
                "agent.multi_turn": not self.agent.one_shot,
                "agent.stream": True,
            },
        ) as sp:
            trimmed = self._trim_history()
            buffer: list[str] = []
            try:
                for piece in self.agent.respond_stream(trimmed):
                    if not piece:
                        continue
                    buffer.append(piece)
                    yield piece
            except Exception as exc:  # noqa: BLE001
                sp.set_attribute("error", True)
                sp.set_attribute("error.type", type(exc).__name__)
                logger.error("流式生成失败：%s", exc)
                self.error = f"{type(exc).__name__}: {exc}"
                hint = "未配置 OPENAI_API_KEY" if "OPENAI_API_KEY" in str(exc) else "模型调用失败"
                fallback = f"（{hint}，请检查 .env 配置后重试。错误详情：{type(exc).__name__}）"
                buffer.append(fallback)
                yield fallback

            reply = safe_output("".join(buffer))
            sp.set_attribute("agent.reply_chars", len(reply))
            self.history.append(AIMessage(content=reply))
            self.record.append(f"\n**助手**：{reply}\n")

    def _trim_history(self) -> list:
        """上下文裁剪：只保留最近 N 条消息，控制 token 增长。

        用户档案（岗位 JD + 关联简历）以 SystemMessage **前置**注入，且放在裁剪之后 ——
        保证即使历史被 `trim_messages` 裁掉，档案也不会丢失。
        这里是生成的唯一入口（普通生成、流式生成、流式线程都走它），改一处即可全局生效。
        """
        if len(self.history) <= config.MAX_HISTORY:
            trimmed = list(self.history)
        else:
            trimmed = list(
                trim_messages(
                    self.history,
                    max_tokens=config.MAX_HISTORY,
                    strategy="last",
                    token_counter=len,
                    start_on="human",
                    allow_partial=False,
                )
            )
        if self.user_context:
            trimmed.insert(0, SystemMessage(content=self.user_context))
        return trimmed

    # ------------------------------------------------------------ 只读属性
    @property
    def label(self) -> str:
        return MODE_LABELS.get(self.mode, self.mode)

    @property
    def citations(self) -> list[str]:
        return getattr(self.agent, "citations", []) or []

    @property
    def multi_turn(self) -> bool:
        return not self.agent.one_shot

    def summary(self) -> str:
        return summarize(" ".join(self.record), 200)
