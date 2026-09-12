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
ENABLE_CACHE: bool = _bool("ENABLE_CACHE", True)

# ---------------------------------------------------------------- Function Calling 自主工具调用
# 关闭（默认）时完全走原有「显式检索」路径，行为与改造前逐字节一致，基线用例零改动；
# 开启后模型可在单轮内自主决定调用联网 / 知识库工具，多步推理后再作答。
# 两条路径可通过本开关切换，构成 A/B 对比实验的两组。
ENABLE_TOOL_CALLING: bool = _bool("ENABLE_TOOL_CALLING", False)
# 工具调用轮次硬上限：达到上限强制收束为自然语言作答，避免不可控的模型往返膨胀。
TOOL_CALLING_MAX_STEPS: int = _int("TOOL_CALLING_MAX_STEPS", 3)
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
        "工具调用": ENABLE_TOOL_CALLING,
        "缓存": ENABLE_CACHE,
        "召回缓存": ENABLE_RETRIEVAL_CACHE,
        "Prompt缓存": ENABLE_PROMPT_CACHE,
    }
