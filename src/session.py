"""会话管理：替代原 Notebook 中 `while True + input()` 的阻塞循环。

驱动模型：UI / API 每次收到一条用户输入，就调用 `step()` 推进一步。
Agent 内部不再持有循环，因此同一套逻辑可同时服务 Streamlit 与 FastAPI。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from . import config
from .agents import create_agent
from .logging_setup import get_logger, summarize
from .model_routing import select_model
from .rag.pipeline import RetrievalConfig
from .safety import check_high_risk, check_input, safe_output
from .state import MODE_LABELS
from .storage import DRAFT_SUFFIX, save_file
from .telemetry import span

logger = get_logger(__name__)

# ---------------------------------------------------------------- 槽位抽取规则（P2-6）
# 只抽「可枚举的硬约束」：城市 / 岗位方向 / 时间范围 / 学历。
# 刻意不去理解任意语义——规则命中的部分准确、可复现、零成本，
# 剩下开放语义交给模型自己从对话里读（这也是它默认关闭、需要 A/B 验证的原因）。
SLOT_CITY_KEYWORDS: tuple[str, ...] = (
    "北京", "上海", "广州", "深圳", "杭州", "长沙", "成都", "武汉", "南京", "西安",
    "苏州", "合肥", "厦门", "天津", "重庆", "郑州", "青岛", "南昌", "福州", "远程", "remote",
)

SLOT_PATTERNS: dict[str, str] = {
    # 允许一个前导英文限定词（如「Java 后端工程师」里的 Java），否则按空格切分会丢信息
    "岗位方向": r"((?:[A-Za-z]+\s+)?[\u4e00-\u9fa5]{2,10}(?:工程师|开发|算法|产品经理|研究员|测试|运维))",
    "时间范围": r"(秋招|春招|校招|社招|暑期实习|\d{4}\s*年(?:\s*\d{1,2}\s*月)?|\d{1,2}\s*个?月内?)",
    "学历": r"(博士|硕士|研究生|本科|大专)",
}


def _user_context_message(text: str) -> SystemMessage:
    """将 user_context 注入为 SystemMessage。

    开启且模型支持（qwen 系列）且文本足够长时，标记 `cache_control` 让 DashScope
    对这段稳定前缀做上下文缓存，跳过重复 prefill；否则返回普通字符串 SystemMessage，
    行为与改动前完全一致。
    """
    enabled = (
        config.ENABLE_PROMPT_CACHE
        and config.MODEL_NAME.lower().startswith(config.PROMPT_CACHE_MODELS)
        and len(text) >= config.PROMPT_CACHE_MIN_CHARS
    )
    if enabled:
        return SystemMessage(
            content=[
                {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}
            ]
        )
    return SystemMessage(content=text)


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
        # 把「用户档案是否可用」直接交给 Agent，供 JD 匹配 / 面试复盘等场景
        # 在缺少岗位与简历时给出明确降级提示（而非凭空编造结论）。
        self.agent.extra_context = self.user_context
        # ---- 工具调用过程（Function Calling 路径）----
        # Agent 每次调用工具都会向此列表追加一条过程事件，供 API / UI 按序透出，
        # 让「模型正在调用什么工具」对用户可见；默认（ENABLE_TOOL_CALLING=false）下
        # 该列表始终为空，既有行为零变化。
        self.tool_events: list[dict] = []
        self.agent.tool_event_listener = self.tool_events.append
        # 会话标识（由 API 层注入）：用于产物幂等命名，以及作为单轮工具闭环的 checkpoint 隔离键
        self.session_id: str = ""
        # 槽位记忆（P2-6）：关键约束单独存一份，与「聊天历史」解耦
        self.slots: dict[str, str] = {}
        self.slot_updates: list[str] = []
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

    # ------------------------------------------------------------ 恢复（P1-4）
    def restore(self, messages: Sequence[tuple[str, str]]) -> None:
        """用持久化的消息重建会话（进程重启后的恢复入口）。

        恢复范围刻意限定在「会话级状态」：历史消息 + 转写记录（产物正文由它拼装）。
        不去假装恢复「执行中那一轮」的图状态——那是 checkpoint 的职责，
        两者职责不重叠，也就不需要互相猜测对方的状态格式。
        """
        for role, content in messages:
            text = content or ""
            if role == "user":
                self.history.append(HumanMessage(content=text))
                self.record.append(f"**用户**：{text}\n")
            elif role == "assistant":
                self.history.append(AIMessage(content=text))
                self.record.append(f"\n**助手**：{text}\n")
        self.started = bool(self.history)
        if config.ENABLE_SLOT_MEMORY:
            self._update_slots(self.history)  # 恢复的对话同样要能把约束找回来

    @property
    def turn_index(self) -> int:
        """已完成的轮数（按助手回复条数计）。

        用它当 checkpoint 的 thread 键有个关键好处：一轮**还在执行中**时索引不变，
        于是「进程中途挂掉后带着同一 session 重发」会命中同一个 checkpoint 续跑；
        而一轮正常结束后索引递增，下一轮一定是全新 thread，不会误撞上一轮的旧结果。
        """
        return sum(1 for m in self.history if isinstance(m, AIMessage))

    @property
    def tool_thread_id(self) -> str | None:
        """单轮工具闭环的 checkpoint 隔离键；没有会话 id（如本地 Streamlit 用法）时为 None。"""
        return f"{self.session_id}:t{self.turn_index}" if self.session_id else None

    # ------------------------------------------------------------ 生命周期
    def reset_tool_events(self) -> None:
        """开启新一轮前清空过程事件。

        工具调用时间线只描述**当前这一轮**：若不清空，多轮会话会把上一轮的
        步骤混进来，界面看起来像「这一轮调用了很多工具」，与事实不符。
        """
        self.tool_events.clear()

    def start(self, user_query: str) -> str:
        """首轮：准备上下文 -> 生成 -> 一次性模式自动收尾。"""
        self.reset_tool_events()
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

        self.reset_tool_events()
        ok, reason = check_input(user_text)
        if not ok:
            return f"（{reason}）"

        self._detect_high_risk(user_text)

        self.record.append(f"\n**用户**：{user_text}\n")
        self.history.append(HumanMessage(content=user_text))
        return self._generate()

    def start_stream(self, user_query: str, *, auto_finish: bool = True):
        """首轮（流式版）：与 start() 流程一致，只是逐段产出文本。

        `auto_finish=False` 供 API 流式网关使用：API 侧由用户显式点击「导出 / 确认」
        才收尾，因此这里只推进一轮，不自动落盘。除此之外流程与默认完全一致，
        保证「同一套会话逻辑同时服务 Streamlit 与 API」这一约定不被破坏。
        """
        self.reset_tool_events()
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
        if auto_finish and self.agent.one_shot:
            self.finish()

    def step_stream(self, user_text: str, *, auto_finish: bool = True):
        """后续轮（流式版）：与 step() 流程一致，只是逐段产出文本。"""
        if self.finished:
            yield "本轮会话已结束并导出。如需继续，请点击「新建会话」。"
            return
        if not self.started:
            yield from self.start_stream(user_text, auto_finish=auto_finish)
            return

        self.reset_tool_events()
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
            self.draft_path = save_file(
                content, f"{self.agent.artifact}{DRAFT_SUFFIX}", key=self.session_id or None
            )
            self.awaiting_confirmation = True
            logger.info("草稿已产出，等待人工确认 | mode=%s | %s", self.mode, self.draft_path)
            return self.draft_path

        self.artifact_path = save_file(content, self.agent.artifact, key=self.session_id or None)
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
        self.artifact_path = save_file(
            self._build_content(), self.agent.artifact, key=self.session_id or None
        )
        self.finished = True
        self.awaiting_confirmation = False
        logger.info("已人工确认定稿 | mode=%s | %s", self.mode, self.artifact_path)
        return self.artifact_path

    # ------------------------------------------------------------ 内部
    def _max_tokens_for(self) -> int | None:
        """按当前 mode 取输出 token 上限（控制生成长度 9.2.A）。

        开关关闭时返回 None，使 get_chat_model 不传 max_tokens，行为零回归；
        开关开启时优先用 mode 覆盖值，否则全局兜底 MAX_TOKENS。
        """
        if not config.ENABLE_MAX_TOKENS:
            return None
        return config.MAX_TOKENS_BY_MODE.get(self.mode, config.MAX_TOKENS)

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
            model = select_model(
                query=trimmed[-1].content if trimmed else "",
                mode=self.mode,
                user_context=self.user_context,
            )
            max_tokens = self._max_tokens_for()
            try:
                reply = self.agent.respond(
                    trimmed, model=model, max_tokens=max_tokens, thread_id=self.tool_thread_id
                )
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
            model = select_model(
                query=trimmed[-1].content if trimmed else "",
                mode=self.mode,
                user_context=self.user_context,
            )
            max_tokens = self._max_tokens_for()
            buffer: list[str] = []
            try:
                for piece in self.agent.respond_stream(
                    trimmed, model=model, max_tokens=max_tokens, thread_id=self.tool_thread_id
                ):
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
            trimmed.insert(0, _user_context_message(self.user_context))
        # 槽位记忆（P2-6，默认关闭）：在裁剪**之后**注入，保证「目标城市：广东」这类
        # 硬约束不会随早期发言被裁掉。放在画像之后，语义上是「先背景、再约束」。
        if config.ENABLE_SLOT_MEMORY:
            self._update_slots(trimmed)
            slots_message = self._slots_message()
            if slots_message is not None:
                trimmed.insert(1, slots_message)
        return trimmed

    # ------------------------------------------------------------ 槽位记忆
    def _update_slots(self, messages: Sequence) -> None:
        """从历史里的**用户**发言累积关键约束（新值覆盖旧值）。

        为什么需要它：`trim_messages` 会把早期发言裁掉，「只要广东」这种硬约束
        一旦被裁，后面几轮模型就看不见了——记忆不该依赖"还没被裁掉"。
        为什么用规则而不是再调一次模型：槽位每轮都要重放，多一次模型调用
        既慢又不可复现；规则命中的是可枚举的硬约束，足够用。
        """
        for message in messages:
            if not isinstance(message, HumanMessage):
                continue
            text = str(getattr(message, "content", "") or "")
            for city in SLOT_CITY_KEYWORDS:
                if city in text:
                    self._set_slot("目标城市", city)
            for name, pattern in SLOT_PATTERNS.items():
                match = re.search(pattern, text)
                if match:
                    self._set_slot(name, match.group(1).strip())

    def _set_slot(self, key: str, value: str) -> None:
        """写入槽位；值有变化时留一条变更记录（便于排查"模型为什么这么回答"）。"""
        if not value or self.slots.get(key) == value:
            return
        self.slots[key] = value
        self.slot_updates.append(f"{key}={value}")

    def _slots_message(self) -> SystemMessage | None:
        """把槽位渲染为一条 SystemMessage；无槽位或开关关闭时返回 None。"""
        if not (config.ENABLE_SLOT_MEMORY and self.slots):
            return None
        items = "；".join(f"{k}：{v}" for k, v in self.slots.items())
        return SystemMessage(content=f"【会话约束（每轮都必须遵守）】{items}")

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
