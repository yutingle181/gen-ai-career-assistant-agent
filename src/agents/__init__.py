"""场景 Agent 集合。"""

from .. import metrics
from ..logging_setup import get_logger
from ..rag.pipeline import RetrievalConfig
from ..skills import REGISTRY, register_defaults
from ..state import (
    MODE_INTERVIEW_QUESTIONS,
    MODE_INTERVIEW_REVIEW,
    MODE_JD_MATCH,
    MODE_JOB_SEARCH,
    MODE_KNOWLEDGE,
    MODE_MOCK_INTERVIEW,
    MODE_QA,
    MODE_RESUME,
    MODE_TUTORIAL,
)
from .base import BaseAgent
from .interview import InterviewQuestionsAgent, MockInterviewAgent
from .interview_review import InterviewReviewAgent
from .jd_match import JDMatchAgent
from .jobsearch import JobSearchAgent
from .knowledge import KnowledgeAgent
from .learning import QAAgent, TutorialAgent
from .resume import ResumeAgent

logger = get_logger(__name__)

AGENT_CLASSES = {
    MODE_TUTORIAL: TutorialAgent,
    MODE_QA: QAAgent,
    MODE_RESUME: ResumeAgent,
    MODE_INTERVIEW_QUESTIONS: InterviewQuestionsAgent,
    MODE_MOCK_INTERVIEW: MockInterviewAgent,
    MODE_JOB_SEARCH: JobSearchAgent,
    MODE_KNOWLEDGE: KnowledgeAgent,
    MODE_JD_MATCH: JDMatchAgent,
    MODE_INTERVIEW_REVIEW: InterviewReviewAgent,
}

# 技能注册表（Skills 骨架：注册 / 版本 / 权限）：场景声明集中在那里
# ——「有哪些能力、各自什么版本、需要什么权限」有唯一事实来源。
# `AGENT_CLASSES` 保留为派生视图，既有调用方（含测试）零改动。
register_defaults()
_REGISTERED: dict[str, type[BaseAgent]] = {}
for _mode in REGISTRY.modes():
    _skill = REGISTRY.for_mode(_mode)
    if _skill is not None:
        _REGISTERED[_mode] = _skill.agent_cls
if _REGISTERED != AGENT_CLASSES:
    # 注册表与硬编码表不一致说明有人只改了一处：暴露出来而不是静默取其一
    logger.warning(
        "AGENT_CLASSES 与技能注册表不一致 | 仅表里有 %s | 仅注册表里有 %s",
        sorted(set(AGENT_CLASSES) - set(_REGISTERED)),
        sorted(set(_REGISTERED) - set(AGENT_CLASSES)),
    )


def create_agent(
    mode: str,
    kb_name: str | None = None,
    retrieval_cfg: RetrievalConfig | None = None,
) -> BaseAgent:
    """按模式创建 Agent。

    取自技能注册表：已注册且启用的场景用它的 Agent 类；未注册 / 被禁用 /
    接口不兼容被拒的模式一律回退默认答疑场景 —— 保证路由永远有出口。
    """
    skill = REGISTRY.for_mode(mode)
    if skill is None:
        metrics.incr("skill.fallback")
        logger.warning("模式 %s 没有可用 skill，回退默认答疑场景", mode)
    cls = skill.agent_cls if skill is not None else QAAgent
    return cls(kb_name=kb_name, retrieval_cfg=retrieval_cfg)


__all__ = [
    "BaseAgent",
    "TutorialAgent",
    "QAAgent",
    "ResumeAgent",
    "InterviewQuestionsAgent",
    "MockInterviewAgent",
    "JobSearchAgent",
    "KnowledgeAgent",
    "JDMatchAgent",
    "InterviewReviewAgent",
    "create_agent",
    "AGENT_CLASSES",
]
