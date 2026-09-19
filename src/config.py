"""集中配置：读取 .env，定义目录常量与各模块默认值。

设计要点：
- 所有路径使用 pathlib，避免 Windows 反斜杠转义问题；
- 所有开关与超参集中在这一层，业务代码不直接读 os.environ；
- 提供 `describe()` 用于 UI 展示与日志脱敏。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# override=True：让项目根目录的 .env 优先于系统环境变量，
# 便于在 .env 中覆盖（如替换无效的系统级 OPENAI_API_KEY）。
load_dotenv(override=True)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- 目录
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
SAMPLES_DIR = KNOWLEDGE_DIR / "samples"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "Agent_output"
DB_PATH = DATA_DIR / "app.db"

for _d in (
    DATA_DIR,
    KNOWLEDGE_DIR,
    SAMPLES_DIR,
    INDEX_DIR,
    EVAL_DIR,
    CACHE_DIR,
    OUTPUT_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 大模型
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").strip()
MODEL_NAME: str = os.getenv("MODEL_NAME", "deepseek-chat").strip()
TEMPERATURE: float = _float("TEMPERATURE", 0.5)
REQUEST_TIMEOUT: int = _int("REQUEST_TIMEOUT", 120)
MAX_RETRIES: int = _int("MAX_RETRIES", 3)

# ---------------------------------------------------------------- 分级模型路由（9.2.A）
# 仅当强模型为 qwen 家族（生产 qwen-plus）且开启开关时生效：
# 简单问答 / 闲聊走低成本的 qwen-turbo，长任务 / 复杂编排仍走 qwen-plus。
# 非 qwen 部署（如默认 deepseek-chat）或未开启时，全部回落 MODEL_STRONG，行为零回归。
MODEL_FAST: str = os.getenv("MODEL_FAST", "qwen-turbo").strip()
MODEL_STRONG: str = os.getenv("MODEL_STRONG", MODEL_NAME).strip()
ENABLE_MODEL_ROUTING: bool = _bool("ENABLE_MODEL_ROUTING", True)
# 仅这些 mode 下的「简单」请求才可能降级到 MODEL_FAST
ROUTING_SIMPLE_MODES: tuple[str, ...] = ("tutorial", "qa")
# 超过该长度的 query 视为「复杂」，不降级
ROUTING_SIMPLE_MAX_CHARS: int = _int("ROUTING_SIMPLE_MAX_CHARS", 80)

# ---------------------------------------------------------------- 控制生成长度（9.2.A）
# 为每轮「用户可见」的文本生成设输出 token 上限，避免个别超长生成（如 111s）
# 拖垮首 token 与整体体验。硬上限由模型侧截断，可测可观测；结构化输出 /
# 评测 / rerank 等短链路不强制（保持默认），本项不改动其调用点。
# 关闭开关或上限为 None 时，get_chat_model 不传 max_tokens，行为零回归。
ENABLE_MAX_TOKENS: bool = _bool("ENABLE_MAX_TOKENS", True)
# 全局默认上限（输出 tokens）；未被下方按 mode 覆盖时兜底
MAX_TOKENS: int = _int("MAX_TOKENS", 2048)
# 按 mode 覆盖：聊天答疑类收紧，简历 / 面试长产物类放宽
MAX_TOKENS_BY_MODE: dict[str, int] = {
    "qa": 1500,
    "tutorial": 1500,
    "knowledge": 1500,
    "fallback": 1500,
    "mock_interview": 1500,
    "job_search": 2000,
    "interview_questions": 3000,
    "resume": 3500,
}


# ---------------------------------------------------------------- Embedding
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "api").strip().lower()
EMBEDDING_BASE_URL: str = os.getenv(
    "EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"
).strip()
EMBEDDING_API_KEY: str = (
    os.getenv("EMBEDDING_API_KEY", "").strip() or OPENAI_API_KEY
)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
LOCAL_EMBEDDING_MODEL: str = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
).strip()
EMBEDDING_BATCH_SIZE: int = _int("EMBEDDING_BATCH_SIZE", 32)


# ---------------------------------------------------------------- Rerank
RERANK_PROVIDER: str = os.getenv("RERANK_PROVIDER", "llm").strip().lower()
RERANK_BASE_URL: str = os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1").strip()
RERANK_API_KEY: str = os.getenv("RERANK_API_KEY", "").strip() or EMBEDDING_API_KEY
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3").strip()


# ---------------------------------------------------------------- 检索默认参数
USE_BM25: bool = _bool("USE_BM25", True)
USE_VECTOR: bool = _bool("USE_VECTOR", True)
FUSION: str = os.getenv("FUSION", "rrf").strip().lower()
TOP_K: int = _int("TOP_K", 5)
RERANK_TOP_N: int = _int("RERANK_TOP_N", 20)
CHUNK_SIZE: int = _int("CHUNK_SIZE", 500)
CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 80)

# ---------------------------------------------------------------- 召回结果缓存（9.2.B）
# 缓存「BM25 + 向量 -> RRF 融合」的召回候选（未重排），相同 (知识库 + query + 召回参数)
# 直接复用，跳过重复的 query 向量化（网络调用）+ BM25 + RRF。
# 缓存 key 刻意不含 reranker，使「仅重排方式不同」的多组实验共享同一召回结果。
# 关闭开关或 ENABLE_CACHE=False 时完全不读写缓存，行为与本项前逐字节一致。
ENABLE_RETRIEVAL_CACHE: bool = _bool("ENABLE_RETRIEVAL_CACHE", True)
RETRIEVAL_CACHE_TTL: int = _int("RETRIEVAL_CACHE_TTL", 3600)


# ---------------------------------------------------------------- 其他
MAX_HISTORY: int = _int("MAX_HISTORY", 10)
ENABLE_WEB_SEARCH: bool = _bool("ENABLE_WEB_SEARCH", True)
WEB_SEARCH_TIMEOUT: int = _int("WEB_SEARCH_TIMEOUT", 10)
# 联网检索失败分级（P0-2）：同通道重试次数 / 退避秒数 / 备选通道 / 检索式简化长度。
# 默认值的权衡：2 次重试（合计最多 3 次尝试）× 0.5s 退避，最坏多花约 1s；
# 相对「联网不可用时整轮拿不出东西」，这个代价是值得的。
WEB_SEARCH_RETRY_ATTEMPTS: int = _int("WEB_SEARCH_RETRY_ATTEMPTS", 2)
WEB_SEARCH_RETRY_BACKOFF: float = _float("WEB_SEARCH_RETRY_BACKOFF", 0.5)
# 备选通道：DuckDuckGo 的 backend 维度（auto / text / news）。换通道即换上游接口，
# 因此「同参数重试仍失败」后值得一试；留空或 auto 表示不做换通道这一步。
WEB_SEARCH_ALT_BACKEND: str = os.getenv("WEB_SEARCH_ALT_BACKEND", "text").strip()
# 「换参数」时把检索式截断到多少字符（检索式过长 / 引号过多是空结果的常见原因）
WEB_SEARCH_SIMPLIFY_CHARS: int = _int("WEB_SEARCH_SIMPLIFY_CHARS", 60)
# 数据时效（P0-3）：给工具结果注入抓取时间，供 Prompt 判断资料新鲜度
ENABLE_FETCH_TIME: bool = _bool("ENABLE_FETCH_TIME", True)
# 过期阈值：超过该天数的资料会被工具标注「可能已过期」。
# 为什么由工具算而不是让模型算：模型没有时钟、也不擅长日期减法，
# 「距今 1236 天」这种结论式标注的遵守率明显高于只给一个日期。默认 1 年。
FRESHNESS_STALE_DAYS: int = _int("FRESHNESS_STALE_DAYS", 365)
# 一次检索内资料时间跨度超过该天数时，提示「可能来自不同版本」
FRESHNESS_SPAN_DAYS: int = _int("FRESHNESS_SPAN_DAYS", 180)
# 工具熔断（P2-7）：重试解决「单次抖动」，不解决「上游整体挂了」。
# 连续失败到阈值后，冷却期内直接短路（不再打上游），把成倍的等待时间换成立刻如实告知。
ENABLE_TOOL_CIRCUIT_BREAKER: bool = _bool("ENABLE_TOOL_CIRCUIT_BREAKER", True)
TOOL_BREAKER_THRESHOLD: int = _int("TOOL_BREAKER_THRESHOLD", 3)
TOOL_BREAKER_COOLDOWN: int = _int("TOOL_BREAKER_COOLDOWN", 60)

# ---------------------------------------------------------------- 任务生命周期（取消 / 排队）
# 排队上限：超出的请求立刻 429，而不是无限期挂着（限流按 IP 计数，解决不了
# 「一个客户端并发打满队列」的问题）。
API_MAX_QUEUE: int = _int("API_MAX_QUEUE", 8)
# 单个请求最多排队等待多少秒；超时返回 503，让调用方拿到明确结论而不是一直等
API_QUEUE_TIMEOUT: float = _float("API_QUEUE_TIMEOUT", 60.0)

# ---------------------------------------------------------------- 故障注入（演示 / 评测用）
# 默认关闭。打开后按概率给指定工具注入模拟故障，用来验证「失败分级 / 熔断 / Failure Onset」
# 在真出问题时确实生效——这三块在 happy path 上永远是 0 失败，证明不了任何东西。
# 固定种子保证同一批输入得到同一串故障；并发会打乱调用顺序，注入实验请串行（--workers 1）。
ENABLE_FAULT_INJECTION: bool = _bool("ENABLE_FAULT_INJECTION", False)
FAULT_INJECT_TOOL: str = os.getenv("FAULT_INJECT_TOOL", "knowledge_search").strip()
# 取值：http_5xx / http_4xx / timeout / empty（empty = 正常返回但没结果，与报错语义不同）
FAULT_INJECT_KIND: str = os.getenv("FAULT_INJECT_KIND", "http_5xx").strip().lower()
FAULT_INJECT_RATE: float = _float("FAULT_INJECT_RATE", 0.0)
FAULT_INJECT_SEED: int = _int("FAULT_INJECT_SEED", 42)
ENABLE_CACHE: bool = _bool("ENABLE_CACHE", True)

# ---------------------------------------------------------------- 可恢复性（P1-4）
# 进程挂掉后，同一个 session_id 必须能接着聊：会话级状态（历史 / 转写 / 模式 / 产物）
# 从 SQLite 重建；单轮工具闭环另外挂图级 checkpoint，使「同一轮重试」不重复执行工具。
ENABLE_SESSION_RESUME: bool = _bool("ENABLE_SESSION_RESUME", True)
# 槽位记忆（P2-6）：把「目标城市 / 岗位方向 / 时间范围 / 学历」这类硬约束单独记下来，
# 每轮裁剪后重新注入，避免被 trim_messages 裁掉后模型就"忘了"。
# **默认关闭**：它改变了提示词组装，属于有回归风险的能力——
# 先关着，用 A/B 证明它确实提升约束遵循率再打开（沿用工具调用开关的同一套态度）。
ENABLE_SLOT_MEMORY: bool = _bool("ENABLE_SLOT_MEMORY", False)
# 图级 checkpoint 后端按「sqlite → memory → 不启用」自动降级（见 src/graph/checkpoint.py）：
# 装了 langgraph-checkpoint-sqlite 才是跨进程的；只装 langgraph-checkpoint 则为进程内。
ENABLE_CHECKPOINT: bool = _bool("ENABLE_CHECKPOINT", True)
CHECKPOINT_DB_PATH: str = os.getenv(
    "CHECKPOINT_DB_PATH", str(DATA_DIR / "checkpoints.sqlite")
).strip()

# ---------------------------------------------------------------- Function Calling 自主工具调用
# 关闭（默认）时完全走原有「显式检索」路径，行为与改造前逐字节一致，基线用例零改动；
# 开启后模型可在单轮内自主决定调用联网 / 知识库工具，多步推理后再作答。
# 两条路径可通过本开关切换，构成 A/B 对比实验的两组。
ENABLE_TOOL_CALLING: bool = _bool("ENABLE_TOOL_CALLING", False)
# 工具调用轮次硬上限：达到上限强制收束为自然语言作答，避免不可控的模型往返膨胀。
TOOL_CALLING_MAX_STEPS: int = _int("TOOL_CALLING_MAX_STEPS", 3)
# 图级递归上限（LangGraph recursion_limit，默认 25）。与上面的「单轮轮次上限」是两层：
# 轮次上限管 ReAct 调了几轮工具，这里管整个图被执行了多少个超步。
# 显式配置的意义是：即使条件边判断失效（例如模型持续产出 tool_calls 而收束节点未被走到），
# 图执行也会被硬性终止，并且我们捕获 GraphRecursionError 后强制收束为阶段性结论，
# 而不是把栈抛给用户。
TOOL_RECURSION_LIMIT: int = _int("TOOL_RECURSION_LIMIT", 25)
# ---------------------------------------------------------------- 会话/Prompt 缓存（DashScope 上下文缓存）
# 仅对支持显式缓存的模型（qwen 系列）生效；开启后 user_context 长前缀标记 cache_control，
# 跳过重复 prefill。不支持的模型 / 开关关闭时自动退化为普通消息，行为不回归。
ENABLE_PROMPT_CACHE: bool = _bool("ENABLE_PROMPT_CACHE", True)
# 仅这些模型前缀启用 prompt 缓存（DashScope 上下文缓存要求 qwen 系列）
PROMPT_CACHE_MODELS: tuple[str, ...] = ("qwen",)
# user_context 短于此字符数不标记缓存（显式缓存最小约 1024 token，短前缀无意义）
PROMPT_CACHE_MIN_CHARS: int = _int("PROMPT_CACHE_MIN_CHARS", 1024)
# 评测并发度：跑 A-B 实验时按样本并发的线程数。1 表示串行（便于调试）。
# 调大可显著缩短评测耗时，但会提高对模型/Embedding 接口的并发压力，注意对方限流。
EVAL_MAX_WORKERS: int = _int("EVAL_MAX_WORKERS", 4)
# 人机协同：高风险场景（简历 / 模拟面试 / 职位清单，或命中高风险话题）
# 先产出草稿并标记待确认，人工确认后才落盘定稿。关闭则恢复为全自动直接定稿。
ENABLE_HUMAN_CONFIRM: bool = _bool("ENABLE_HUMAN_CONFIRM", True)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()

# ---------------------------------------------------------------- API 安全
# 业务接口鉴权：留空则关闭（兼容演示）；填入后所有 /chat /knowledge /sessions 需带 X-API-Key。
API_KEY: str = os.getenv("API_KEY", "").strip()
# 速率限制：每客户端 IP 每分钟请求上限；<=0 表示关闭限流。
API_RATE_LIMIT: int = _int("API_RATE_LIMIT", 60)


def mask_key(key: str) -> str:
    """脱敏展示密钥，仅保留前 4 位与后 4 位。"""
    if not key:
        return "(未配置)"
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:4]}{'*' * 6}{key[-4:]}"


def llm_ready() -> bool:
    """大模型是否已配置可用。"""
    return bool(OPENAI_API_KEY)


def embedding_ready() -> bool:
    """Embedding 是否已配置可用（本地模式无需 Key）。"""
    return EMBEDDING_PROVIDER == "local" or bool(EMBEDDING_API_KEY)


def describe() -> dict:
    """供 UI / 日志展示的配置快照（已脱敏）。"""
    return {
        "模型": MODEL_NAME,
        "模型地址": OPENAI_BASE_URL,
        "API Key": mask_key(OPENAI_API_KEY),
        "Embedding": f"{EMBEDDING_PROVIDER} / {EMBEDDING_MODEL}",
        "重排": RERANK_PROVIDER,
        "检索": f"BM25={USE_BM25} 向量={USE_VECTOR} 融合={FUSION} top_k={TOP_K}",
        "切分": f"size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}",
        "联网搜索": ENABLE_WEB_SEARCH,
        "检索重试": f"{WEB_SEARCH_RETRY_ATTEMPTS}次/退避{WEB_SEARCH_RETRY_BACKOFF}s",
        "抓取时间": ENABLE_FETCH_TIME,
        "工具调用": ENABLE_TOOL_CALLING,
        "图级递归上限": TOOL_RECURSION_LIMIT,
        "缓存": ENABLE_CACHE,
        "召回缓存": ENABLE_RETRIEVAL_CACHE,
        "Prompt缓存": ENABLE_PROMPT_CACHE,
    }
