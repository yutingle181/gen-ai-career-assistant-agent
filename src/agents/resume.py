"""简历 Agent：分步收集信息生成简历，支持基于 JD 改写。"""

from __future__ import annotations

import re

from ..state import MODE_RESUME
from .base import BaseAgent

_JD_HINTS = ("jd", "job description", "岗位描述", "职位描述", "招聘要求", "任职要求", "岗位职责")


class ResumeAgent(BaseAgent):
    """简历制作（多轮，4-5 步收集信息）。"""

    mode = MODE_RESUME
    artifact = "Resume"
    one_shot = False
    needs_search = False
    requires_confirmation = True  # 简历会外发给招聘方，定稿前必须人工审阅

    def prepare(self, query: str) -> str:
        """检测到用户贴了 JD 时，追加改写指令。"""
        low = (query or "").lower()
        if any(h in low for h in _JD_HINTS) or len(query or "") > 200:
            return (
                "\n\n【检测到岗位描述】请按人设中的 JD 改写流程工作："
                "先提炼关键词与能力要求，再指出差距，最后给出改写后的简历。"
            )
        return ""

    @staticmethod
    def extract_markdown(text: str) -> str:
        """抽取 Markdown 代码块中的简历正文。"""
        match = re.search(r"```(?:markdown)?\s*(.*?)```", text or "", flags=re.DOTALL)
        return match.group(1).strip() if match else (text or "").strip()
