"""Pydantic 数据模型：结构化输出与检索结果契约。

用途：
1. 替代原 Notebook 中脆弱的字符串判断（如 `if '1' in category`）；
2. 供 `with_structured_output` 使用，落地 JD 要求的「结构化输出」能力。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- 分类
class CategoryResult(BaseModel):
    """一级分类结果。"""

    category: Literal[
        "learning", "resume", "interview", "job_search", "knowledge", "jd_match", "interview_review"
    ] = Field(description="查询所属的一级类别")
    reason: str = Field(default="", description="一句话分类依据，便于排障与界面展示")


class LearningSubResult(BaseModel):
    """学习类二级分类：教程 or 问答。"""

    sub: Literal["tutorial", "question"] = Field(description="教程生成或答疑")


class InterviewSubResult(BaseModel):
    """面试类二级分类：模拟面试 or 真题。"""

    sub: Literal["mock", "question"] = Field(description="模拟面试或面试真题")


# ---------------------------------------------------------------- 检索
class RetrievedChunk(BaseModel):
    """召回的文档片段。"""

    chunk_id: str
    text: str
    source: str = ""
    page: int | None = None
    score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    """回答中引用的来源。"""

    index: int
    source: str
    page: int | None = None
    snippet: str = ""


class RAGAnswer(BaseModel):
    """知识库问答结果。"""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: int = 0
    tokens: int = 0


# ---------------------------------------------------------------- 业务结构化输出
class JobItem(BaseModel):
    """职位条目。"""

    company: str = ""
    title: str = ""
    location: str = ""
    link: str = ""
    source: str = ""


class JobList(BaseModel):
    """职位清单。"""

    jobs: list[JobItem] = Field(default_factory=list)
    summary: str = ""


class InterviewEval(BaseModel):
    """模拟面试评价。"""

    overall: int = Field(default=0, ge=0, le=100, description="总体得分")
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    suggestion: str = ""


class JDMatchResult(BaseModel):
    """JD 匹配诊断结果（界面按「评分卡 + 双栏对照」渲染）。"""

    total_score: int = Field(default=0, ge=0, le=100, description="0-100 总分")
    dimension_scores: dict[str, int] = Field(
        default_factory=dict,
        description="分项得分，键固定为 skills / experience / education / projects",
    )
    matched: list[str] = Field(default_factory=list, description="命中项")
    gaps: list[str] = Field(default_factory=list, description="缺口项")
    interview_focus: list[str] = Field(default_factory=list, description="面试准备重点")
    summary: str = Field(default="", description="一句话结论")


class InterviewReview(BaseModel):
    """面试复盘结果（字段与前端复盘视图对齐，避免第二条重复链路）。"""

    overall: int = Field(default=0, ge=0, le=100, description="总体表现分")
    dimensions: dict[str, int] = Field(default_factory=dict, description="分项得分")
    question_chain: list[str] = Field(default_factory=list, description="追问链还原")
    weaknesses: list[str] = Field(default_factory=list, description="薄弱点")
    suggestions: list[str] = Field(default_factory=list, description="改进动作")
    detail: str = Field(default="", description="整体点评")


class ResumeSection(BaseModel):
    """简历的一个章节。"""

    title: str
    bullets: list[str] = Field(default_factory=list)


class EvalRecord(BaseModel):
    """评测集条目。"""

    question: str
    golden_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str = ""


class MetricResult(BaseModel):
    """单组实验的评测结果。"""

    name: str
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    hallucination_rate: float = 0.0
    avg_latency_ms: float = 0.0
    sample_count: int = 0
    config: dict = Field(default_factory=dict)
