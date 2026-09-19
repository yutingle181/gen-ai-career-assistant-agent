"""API 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(description="用户输入")
    session_id: str | None = Field(default=None, description="会话 ID，为空则新建")
    mode: str | None = Field(default=None, description="手动指定模式，为空则自动路由")
    kb_name: str | None = Field(default=None, description="知识库名称")
    top_k: int | None = Field(default=None)
    reranker: str | None = Field(default=None, description="none | llm | api")
    use_bm25: bool | None = Field(default=None)
    use_vector: bool | None = Field(default=None)
    # 用户档案（岗位 JD + 关联简历），由后端网关注入；为空时行为与未注入完全一致
    user_context: str | None = Field(default=None, description="用户档案上下文，由后端注入")


class ChatResponse(BaseModel):
    session_id: str
    mode: str
    mode_label: str = ""
    message: str
    citations: list[str] = Field(default_factory=list)
    artifact: str | None = None  # 定稿产物路径（草稿态下为 None）
    finished: bool = False
    # ---- 人机协同：以下字段均有默认值，向后兼容既有调用方 ----
    requires_confirmation: bool = False  # 该场景产物是否需人工确认
    awaiting_confirmation: bool = False  # 草稿已产出、正等待确认
    draft_artifact: str | None = None  # 草稿产物路径


class FinishRequest(BaseModel):
    session_id: str


class IngestResponse(BaseModel):
    kb_name: str
    files: int = 0
    documents: int = 0
    chunks: int = 0
    dim: int = 0
    elapsed_ms: int = 0


class KnowledgeInfo(BaseModel):
    name: str
    chunks: int = 0
    dim: int = 0
    sources: int = 0
    has_vector: bool = False
    has_bm25: bool = False


class CancelRequest(BaseModel):
    """取消一轮在途对话。"""

    session_id: str = Field(description="要取消的会话 ID")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    reranker: str = "llm"


class SearchHit(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    score: float = 0.0
    text: str = ""


class HealthResponse(BaseModel):
    status: str
    llm: bool
    llm_detail: str = ""
    embedding: bool
    embedding_detail: str = ""
    knowledge_bases: list[str] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
    # 运行时状态（P1-4 / P2-6 / P2-7）：checkpoint 后端、熔断器、会话恢复与槽位记忆开关。
    # 放在 /health 里的理由：这些开关直接决定「上一轮为什么没检索」「重启后能不能接着聊」，
    # 排查时应该在同一个地方看到，而不是翻三份日志。
    runtime: dict = Field(default_factory=dict)
    # 进程内指标快照（工具失败率 / 超轮次率 / 熔断拦截率等）
    metrics: dict = Field(default_factory=dict)


class CostSummary(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
