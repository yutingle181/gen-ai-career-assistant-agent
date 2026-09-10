"""LangGraph 状态定义。

兼容原 Notebook 的 query / category / response 三字段，并扩展会话与 RAG 字段。
"""

from __future__ import annotations

from typing import TypedDict

# 一级类别
CAT_LEARNING = "learning"
CAT_RESUME = "resume"
CAT_INTERVIEW = "interview"
CAT_JOB_SEARCH = "job_search"
CAT_KNOWLEDGE = "knowledge"
CAT_FALLBACK = "fallback"

# 会话模式
MODE_TUTORIAL = "tutorial"
MODE_QA = "qa"
MODE_RESUME = "resume"
MODE_INTERVIEW_QUESTIONS = "interview_questions"
MODE_MOCK_INTERVIEW = "mock_interview"
MODE_JOB_SEARCH = "job_search"
MODE_KNOWLEDGE = "knowledge"
MODE_FALLBACK = "fallback"

MODE_LABELS = {
    MODE_TUTORIAL: "教程生成",
    MODE_QA: "答疑问答",
    MODE_RESUME: "简历制作",
    MODE_INTERVIEW_QUESTIONS: "面试真题",
    MODE_MOCK_INTERVIEW: "模拟面试",
    MODE_JOB_SEARCH: "职位搜索",
    MODE_KNOWLEDGE: "知识库问答",
    MODE_FALLBACK: "未识别",
}

# 需要多轮交互的模式（一次性生成的模式不在其中）
MULTI_TURN_MODES = {
    MODE_QA,
    MODE_RESUME,
    MODE_INTERVIEW_QUESTIONS,
    MODE_MOCK_INTERVIEW,
    MODE_KNOWLEDGE,
}


class State(TypedDict, total=False):
    """工作流状态。"""

    query: str  # 用户原始输入
    category: str  # 一级/二级分类结果
    response: str  # 最终产物路径或汇总文本
    mode: str  # 会话模式
    message: str  # 本轮返回给用户的消息
    citations: list[str]  # 知识库问答的引用来源
    artifacts: list[str]  # 生成的 Markdown 文件路径
    error: str  # 降级或异常时的中文提示
