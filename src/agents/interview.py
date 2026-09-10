"""面试类 Agent：面试真题（一次性）与模拟面试（多轮 + 结构化评价）。"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import HumanMessage

from ..llm import get_chat_model
from ..logging_setup import get_logger
from ..models import InterviewEval
from ..state import MODE_INTERVIEW_QUESTIONS, MODE_MOCK_INTERVIEW
from .base import BaseAgent

logger = get_logger(__name__)


class InterviewQuestionsAgent(BaseAgent):
    """面试真题：检索后一次性生成题库。"""

    mode = MODE_INTERVIEW_QUESTIONS
    artifact = "Interview_questions"
    one_shot = True
    needs_search = True


class MockInterviewAgent(BaseAgent):
    """模拟面试：一问一答，结束时用结构化输出给出面评。"""

    mode = MODE_MOCK_INTERVIEW
    artifact = "Mock_Interview"
    one_shot = False
    needs_search = False
    requires_confirmation = True  # 面评结论直接影响候选人判断，需人工确认后再给出

    EVAL_SYSTEM = (
        "你是一位资深 GenAI 面试官。下面是刚才的模拟面试记录，"
        "请对候选人给出结构化评价：总体得分（0-100）、3 条优点、3 条改进建议、"
        "以及一段总结性建议。评价要具体，指出回答中暴露的知识薄弱点。"
    )

    def finalize(self, transcript: str) -> str:
        if not transcript.strip():
            return ""
        evaluation = self._structured_eval(transcript)
        if evaluation is None:
            return ""
        return (
            "\n\n---\n\n## 面试评价\n\n"
            f"**总体得分：{evaluation.overall} / 100**\n\n"
            "### 优点\n"
            + "\n".join(f"- {s}" for s in evaluation.strengths)
            + "\n\n### 改进建议\n"
            + "\n".join(f"- {s}" for s in evaluation.improvements)
            + f"\n\n### 总结\n{evaluation.suggestion}\n"
        )

    def _structured_eval(self, transcript: str) -> InterviewEval | None:
        """用 with_structured_output 落地「结构化输出」能力，失败则回退。"""
        try:
            model = get_chat_model(temperature=0).with_structured_output(InterviewEval)
            result = model.invoke(
                [HumanMessage(content=f"{self.EVAL_SYSTEM}\n\n面试记录：\n{transcript[:6000]}")]
            )
            if isinstance(result, InterviewEval):
                return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("结构化面评失败，回退为普通生成：%s", exc)
        return None

    def respond(self, history: Sequence) -> str:
        return super().respond(history)
