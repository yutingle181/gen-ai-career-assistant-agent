"""学习类 Agent：教程生成（一次性）与答疑（多轮）。"""

from __future__ import annotations

from ..state import MODE_QA, MODE_TUTORIAL
from .base import BaseAgent


class TutorialAgent(BaseAgent):
    """教程生成：联网检索后一次性产出 Markdown 教程。"""

    mode = MODE_TUTORIAL
    artifact = "Tutorial"
    one_shot = True
    needs_search = True


class QAAgent(BaseAgent):
    """答疑问答：资深 GenAI 工程师人设，多轮追问。"""

    mode = MODE_QA
    artifact = "Q&A_Doubt_Session"
    one_shot = False
    needs_search = False
