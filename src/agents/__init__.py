"""场景 Agent 集合。"""

from ..rag.pipeline import RetrievalConfig
from ..state import (
    MODE_INTERVIEW_QUESTIONS,
    MODE_JOB_SEARCH,
    MODE_KNOWLEDGE,
    MODE_MOCK_INTERVIEW,
    MODE_QA,
    MODE_RESUME,
    MODE_TUTORIAL,
)
from .base import BaseAgent
from .interview import InterviewQuestionsAgent, MockInterviewAgent
from .jobsearch import JobSearchAgent
from .knowledge import KnowledgeAgent
from .learning import QAAgent, TutorialAgent
from .resume import ResumeAgent

AGENT_CLASSES = {
    MODE_TUTORIAL: TutorialAgent,
    MODE_QA: QAAgent,
    MODE_RESUME: ResumeAgent,
    MODE_INTERVIEW_QUESTIONS: InterviewQuestionsAgent,
    MODE_MOCK_INTERVIEW: MockInterviewAgent,
    MODE_JOB_SEARCH: JobSearchAgent,
    MODE_KNOWLEDGE: KnowledgeAgent,
}


def create_agent(
    mode: str,
    kb_name: str | None = None,
    retrieval_cfg: RetrievalConfig | None = None,
) -> BaseAgent:
    """按模式创建 Agent。"""
    cls = AGENT_CLASSES.get(mode, QAAgent)
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
    "create_agent",
    "AGENT_CLASSES",
]
