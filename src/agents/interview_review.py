"""面试复盘 Agent：把"已经结束的面试"加工成结构化复盘报告。

定位说明：jobseeker 侧已有「会话归档 -> 复盘字段」的存储链路，但归档内容
未被模型加工过。本 Agent 补上这段：产出结构化的评分 / 追问链 / 薄弱点 /
改进动作，字段与前端复盘视图对齐，避免出现第二条重复链路。
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from ..llm import get_chat_model, llm_invoke
from ..logging_setup import get_logger
from ..models import InterviewReview
from ..state import MODE_INTERVIEW_REVIEW
from .base import BaseAgent

logger = get_logger(__name__)

MISSING_CONTEXT_TIP = (
    "【提示】还没有可复盘的面试记录。可以把面试中的问题与你的回答贴给我"
    "（或先在「模拟面试」里完成一场），我再做复盘。"
)


def render_interview_review(review: InterviewReview) -> str:
    """把结构化复盘渲染为 Markdown（供对话展示与产物落盘复用）。"""
    lines = [f"## 面试复盘：{review.overall}/100", ""]
    if review.detail:
        lines += [review.detail, ""]

    if review.dimensions:
        lines.append("### 分项得分")
        for key, score in review.dimensions.items():
            lines.append(f"- {key}：{score}/100")
        lines.append("")

    if review.question_chain:
        lines.append("### 追问链还原")
        lines += [f"{i + 1}. {step}" for i, step in enumerate(review.question_chain)]
        lines.append("")

    if review.weaknesses:
        lines.append("### 薄弱点")
        lines += [f"- {item}" for item in review.weaknesses]
        lines.append("")

    if review.suggestions:
        lines.append("### 改进动作")
        lines += [f"- [ ] {item}" for item in review.suggestions]

    return "\n".join(lines).strip()


class InterviewReviewAgent(BaseAgent):
    """面试复盘：一次产出完整复盘报告。"""

    mode = MODE_INTERVIEW_REVIEW
    artifact = "Interview_Review"
    one_shot = True
    requires_confirmation = False

    def __init__(self, kb_name: str | None = None, retrieval_cfg=None) -> None:
        super().__init__(kb_name=kb_name, retrieval_cfg=retrieval_cfg)
        self.result: InterviewReview | None = None

    def respond(self, history: Sequence[BaseMessage], model: str | None = None,
                max_tokens: int | None = None, thread_id: str | None = None) -> str:
        """产出结构化复盘；结构化输出不可用时回退为纯文本点评。"""
        if self.use_tool_calling:
            return super().respond(
                history, model=model, max_tokens=max_tokens, thread_id=thread_id
            )

        messages = self.build_messages(history)
        try:
            llm = get_chat_model(temperature=0, model=model, max_tokens=max_tokens)
            result = llm.with_structured_output(InterviewReview).invoke(messages)
            if isinstance(result, InterviewReview):
                self.result = result
                return self._degrade(render_interview_review(result))
            logger.warning("面试复盘结构化输出类型异常：%s", type(result).__name__)
        except Exception as exc:  # noqa: BLE001
            logger.warning("面试复盘结构化输出失败，回退为纯文本：%s", exc)
        return self._degrade(llm_invoke(messages, model=model, max_tokens=max_tokens))

    def _degrade(self, text: str) -> str:
        """没有可复盘记录时前置缺口提示，避免模型编造面试过程。"""
        if self.extra_context:
            return text
        return f"{MISSING_CONTEXT_TIP}\n\n{text}"

    def respond_stream(self, history: Sequence[BaseMessage], model: str | None = None,
                       max_tokens: int | None = None, thread_id: str | None = None):
        """一次产出后分段推送。"""
        text = self.respond(history, model=model, max_tokens=max_tokens, thread_id=thread_id)
        step = 40
        for i in range(0, len(text), step):
            yield text[i : i + step]

    def finalize(self, transcript: str) -> str:
        if self.result is not None:
            return render_interview_review(self.result)
        return transcript[-4000:]
