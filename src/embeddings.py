"""Embedding 抽象层：默认云端 OpenAI 兼容 /embeddings，可选本地 HuggingFace。

选型理由：
- 本地 bge-m3 需 torch（1-2GB）且 CPU 推理慢，对无独显机器不友好；
- 云端接口 + faiss-cpu 可在 10 分钟内装完并跑通；
- 通过 EmbeddingFactory 抽象，两者可一行配置切换，业务代码零改动。
"""

from __future__ import annotations

from collections.abc import Sequence

from . import config
from .logging_setup import get_logger

logger = get_logger(__name__)


class BaseEmbedder:
    """Embedding 统一接口。"""

    name: str = "base"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @property
    def dim(self) -> int:
        raise NotImplementedError


class APIEmbedder(BaseEmbedder):
    """云端 OpenAI 兼容 /embeddings 接口。"""

    name = "api"

    def __init__(self) -> None:
        if not config.EMBEDDING_API_KEY:
            raise RuntimeError(
                "未配置 EMBEDDING_API_KEY，且未回退到 OPENAI_API_KEY。"
                "请在 .env 中补充 Embedding 密钥，或设置 EMBEDDING_PROVIDER=local。"
            )
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.EMBEDDING_API_KEY,
            base_url=config.EMBEDDING_BASE_URL,
            check_embedding_ctx_length=False,  # 非 tiktoken 模型不做上下文长度预检
            chunk_size=config.EMBEDDING_BATCH_SIZE,
        )
        self._dim: int | None = None
        logger.info(
            "Embedding 初始化 | provider=api model=%s base=%s",
            config.EMBEDDING_MODEL,
            config.EMBEDDING_BASE_URL,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        clean = [t if t.strip() else " " for t in texts]
        return self._client.embed_documents(list(clean))

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("维度探测"))
        return self._dim


class LocalEmbedder(BaseEmbedder):
    """本地 HuggingFace Embedding（可选依赖，需 sentence-transformers + torch）。"""

    name = "local"

    def __init__(self) -> None:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "本地 Embedding 需要可选依赖，请执行：pip install -r requirements-optional.txt"
            ) from exc

        self._client = HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._dim: int | None = None
        logger.info("Embedding 初始化 | provider=local model=%s", config.LOCAL_EMBEDDING_MODEL)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        clean = [t if t.strip() else " " for t in texts]
        return self._client.embed_documents(list(clean))

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("维度探测"))
        return self._dim


_EMBEDDER: BaseEmbedder | None = None


def get_embedder(force: bool = False) -> BaseEmbedder:
    """获取（并缓存）Embedder 实例。"""
    global _EMBEDDER
    if _EMBEDDER is None or force:
        provider = config.EMBEDDING_PROVIDER
        if provider == "local":
            _EMBEDDER = LocalEmbedder()
        elif provider == "api":
            _EMBEDDER = APIEmbedder()
        else:
            logger.warning("未知的 EMBEDDING_PROVIDER=%s，回退为 api", provider)
            _EMBEDDER = APIEmbedder()
    return _EMBEDDER


def check_embedding_health() -> tuple[bool, str]:
    """探测 Embedding 可用性，返回 (是否可用, 说明)。"""
    try:
        emb = get_embedder()
        vec = emb.embed_query("健康检查")
        model = config.EMBEDDING_MODEL if emb.name == "api" else config.LOCAL_EMBEDDING_MODEL
        return True, f"{emb.name}/{model} dim={len(vec)}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
