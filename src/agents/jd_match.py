"""JD 匹配诊断 Agent：把岗位 JD 与简历放在一起，产出可解释的匹配评分。

产物形态是「评分卡 + 命中/缺口双栏 + 面试准备重点」——结构化输出保证界面
能稳定渲染分项进度条，而不是让模型自由发挥一段散文。
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from ..llm import get_chat_model, llm_invoke
from ..logging_setup import get_logger
from ..models import JDMatchResult
from ..state import MODE_JD_MATCH
from .base import BaseAgent

logger = get_logger(__name__)

# 评分卡的固定维度：键名固定，前端才能稳定渲染分项进度条
DIMENSIONS = ("skills", "experience", "education", "projects")

# 提示词里已固定英文键名，但模型偶尔会「本地化」键名（skills -> 技能），
# 这里补一层别名，保证界面标题始终是干净的中文维度名，而不是裸 key。
DIMENSION_LABELS = {
    "skills": "技能匹配",
    "skill": "技能匹配",
    "技能": "技能匹配",
    "技能匹配": "技能匹配",
    "experience": "经验匹配",
    "经验": "经验匹配",
    "经验匹配": "经验匹配",
    "education": "学历匹配",
    "学历": "学历匹配",
    "学历匹配": "学历匹配",
    "projects": "项目匹配",
    "project": "项目匹配",
    "项目": "项目匹配",
    "项目匹配": "项目匹配",
}

MISSING_CONTEXT_TIP = (
    "【提示】尚未关联岗位与简历。可在提问里直接粘贴 JD 与简历正文，"
    "或先在「职位」「简历」页面完成关联，我再给出匹配诊断。"
)


def render_jd_match(result: JDMatchResult) -> str:
    """把结构化诊断结果渲染为 Markdown 报告（供对话展示与产物落盘复用）。"""
    lines = [f"## JD 匹配诊断：{result.total_score}/100", ""]
    if result.summary:
        lines += [result.summary, ""]

    if result.dimension_scores:
        lines.append("### 分项得分")
        for key, score in result.dimension_scores.items():
            label = DIMENSION_LABELS.get(key, key)
            lines.append(f"- {label}：{score}/100")
        lines.append("")

    if result.matched:
        lines.append("### 命中项")
        lines += [f"- {item}" for item in result.matched]
        lines.append("")

    if result.gaps:
        lines.append("### 缺口项")
        lines += [f"- {item}" for item in result.gaps]
        lines.append("")

    if result.interview_focus:
        lines.append("### 面试准备重点")
        lines += [f"- {item}" for item in result.interview_focus]

    return "\n".join(lines).strip()


class JDMatchAgent(BaseAgent):
    """JD 匹配诊断：一次产出完整报告。"""

    mode = MODE_JD_MATCH
    artifact = "JD_Match"
    one_shot = True  # 一次产出完整诊断报告，不再多轮追问
    requires_confirmation = True  # 匹配结论直接影响求职决策，需人工确认后定稿

    def __init__(self, kb_name: str | None = None, retrieval_cfg=None) -> None:
        super().__init__(kb_name=kb_name, retrieval_cfg=retrieval_cfg)
        self.result: JDMatchResult | None = None

    def respond(self, history: Sequence[BaseMessage], model: str | None = None,
                max_tokens: int | None = None, thread_id: str | None = None) -> str:
        """产出结构化诊断；结构化输出不可用时回退为纯文本，绝不空手而归。"""
        if self.use_tool_calling:
            return super().respond(
                history, model=model, max_tokens=max_tokens, thread_id=thread_id
            )

        messages = self.build_messages(history)
        try:
            llm = get_chat_model(temperature=0, model=model, max_tokens=max_tokens)
            result = llm.with_structured_output(JDMatchResult).invoke(messages)
            if isinstance(result, JDMatchResult):
                self.result = result
                self.citations = []
                return self._degrade(render_jd_match(result))
            logger.warning("JD 匹配结构化输出类型异常：%s", type(result).__name__)
        except Exception as exc:  # noqa: BLE001
            logger.warning("JD 匹配结构化输出失败，回退为纯文本：%s", exc)
        return self._degrade(llm_invoke(messages, model=model, max_tokens=max_tokens))

    def _degrade(self, text: str) -> str:
        """未关联岗位与简历时前置缺口提示，避免模型给出无依据的分数。"""
        if self.extra_context:
            return text
        return f"{MISSING_CONTEXT_TIP}\n\n{text}"

    def respond_stream(self, history: Sequence[BaseMessage], model: str | None = None,
                       max_tokens: int | None = None, thread_id: str | None = None):
        """一次产出后分段推送（评分卡本身是整体，逐字流式收益有限）。"""
        text = self.respond(history, model=model, max_tokens=max_tokens, thread_id=thread_id)
        step = 40
        for i in range(0, len(text), step):
            yield text[i : i + step]

    def finalize(self, transcript: str) -> str:
        """落盘产物优先用渲染好的报告，保证产物可直接阅读。"""
        if self.result is not None:
            return render_jd_match(self.result)
        return transcript[-4000:]
