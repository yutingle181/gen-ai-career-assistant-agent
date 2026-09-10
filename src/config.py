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


# ---------------------------------------------------------------- 其他
MAX_HISTORY: int = _int("MAX_HISTORY", 10)
ENABLE_WEB_SEARCH: bool = _bool("ENABLE_WEB_SEARCH", True)
WEB_SEARCH_TIMEOUT: int = _int("WEB_SEARCH_TIMEOUT", 10)
ENABLE_CACHE: bool = _bool("ENABLE_CACHE", True)
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
        "缓存": ENABLE_CACHE,
    }
