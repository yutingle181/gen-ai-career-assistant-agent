"""职位搜索 Agent：联网检索 + 结构化输出职位清单。"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_chat_model
from ..logging_setup import get_logger
from ..models import JobList
from ..state import MODE_JOB_SEARCH
from .base import BaseAgent

logger = get_logger(__name__)


class JobSearchAgent(BaseAgent):
    """职位搜索：一次性检索并整理为 Markdown 表格。"""

    mode = MODE_JOB_SEARCH
    artifact = "Job_search"
    one_shot = True
    needs_search = True
    # 允许 Function Calling 路径：由模型自主决定检索什么关键词、检索几次。
    # 开关关闭时完全走原有「前置联网检索 + 一次结构化整理」链路。
    needs_tools = True
    requires_confirmation = True  # 职位清单含外链与求职决策建议，需人工确认

    MISSING_INFO_TIP = (
        "为了更准确地检索，请在提问中补充：**目标城市** 与 **岗位名称**"
        "（例如：『长沙 AI 应用工程师 岗位』）。我先按现有信息检索一次。"
    )

    def prepare(self, query: str) -> str:
        context = super().prepare(query)
        has_city = any(
            kw in query
            for kw in ("市", "北京", "上海", "广州", "深圳", "杭州", "长沙", "成都", "武汉", "remote", "远程")
        )
        if not has_city:
            context = f"{self.MISSING_INFO_TIP}\n{context}"
        return context

    def build_messages(self, history: Sequence) -> Sequence:
        """职位整理阶段强制输出结构化表格。"""
        messages = list(super().build_messages(history))
        messages.append(
            HumanMessage(
                content="请以 Markdown 表格输出职位清单，列包含：公司 / 职位 / 地点 / 关键要求 / 链接。"
            )
        )
        return messages

    def finalize(self, transcript: str) -> str:
        """额外产出一份结构化职位 JSON（演示结构化输出能力）。"""
        jobs = self._structured_jobs(transcript)
        if not jobs:
            return ""
        lines = ["\n\n---\n\n## 结构化职位数据（Pydantic 解析结果）\n", "| 公司 | 职位 | 地点 |", "| --- | --- | --- |"]
        for job in jobs.jobs:
            lines.append(f"| {job.company} | {job.title} | {job.location} |")
        if jobs.summary:
            lines.append(f"\n> {jobs.summary}")
        return "\n".join(lines) + "\n"

    def _structured_jobs(self, transcript: str) -> JobList | None:
        try:
            model = get_chat_model(temperature=0).with_structured_output(JobList)
            result = model.invoke(
                [
                    SystemMessage(
                        content="从下面的职位信息中提取结构化条目，尽量保留原文的公司名与职位名。"
                    ),
                    HumanMessage(content=transcript[:6000]),
                ]
            )
            if isinstance(result, JobList):
                return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("结构化职位解析失败，跳过：%s", exc)
        return None
