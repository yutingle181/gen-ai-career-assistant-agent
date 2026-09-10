"""Embedding 层单元测试（离线、确定性）。

用 stub 替换 OpenAIEmbeddings / HuggingFaceEmbeddings，覆盖 api/local 两种 provider、
未知 provider 降级回退、空串占位、健康检查成功/失败等分支，不依赖真实网络或 Key。
"""

from __future__ import annotations

import pytest

from src import config
from src.embeddings import (
    APIEmbedder,
    BaseEmbedder,
    LocalEmbedder,
    check_embedding_health,
    get_embedder,
)


class _StubEmbedder(BaseEmbedder):
    def __init__(self, name: str = "api") -> None:
        self.name = name

    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]

    @property
    def dim(self) -> int:
        return 2


def test_api_embedder_requires_key(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError):
        APIEmbedder()


def test_local_embedder_requires_deps(monkeypatch):
    try:
        import langchain_community.embeddings  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError):
            LocalEmbedder()
    else:
        pytest.skip("sentence-transformers 已安装，无法验证缺失依赖分支")


def test_base_embedder_protocol():
    """BaseEmbedder 默认 embed_query 走 embed_documents，dim 为接口。"""

    class Fake(BaseEmbedder):
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        @property
        def dim(self) -> int:
            return 2

    f = Fake()
    assert f.embed_query("x") == [1.0, 0.0]
    assert f.dim == 2


def test_api_embedder_embed_documents_pads_empty(monkeypatch):
    """空串应替换为空格占位，避免向量接口报空。"""
    emb = APIEmbedder.__new__(APIEmbedder)

    class _Client:
        def embed_documents(self, texts):
            return [[0.1]] * len(texts)

    emb._client = _Client()
    emb._dim = None
    captured: dict = {}
    orig = emb._client.embed_documents

    def _wrap(texts):
        captured["texts"] = list(texts)
        return orig(texts)

    emb._client.embed_documents = _wrap
    out = emb.embed_documents(["hello", ""])
    assert captured["texts"] == ["hello", " "]
    assert len(out) == 2


def test_get_embedder_dispatch(monkeypatch):
    import src.embeddings as emb

    monkeypatch.setattr(emb, "APIEmbedder", lambda: _StubEmbedder("api"))
    monkeypatch.setattr(emb, "LocalEmbedder", lambda: _StubEmbedder("local"))
    monkeypatch.setattr(emb, "_EMBEDDER", None)

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "local")
    assert get_embedder(force=True).name == "local"

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "api")
    assert get_embedder(force=True).name == "api"

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bogus")
    assert get_embedder(force=True).name == "api"  # 未知 provider 降级回退为 api


def test_api_embedder_init_and_usage(monkeypatch):
    """真实走 APIEmbedder.__init__（stub OpenAIEmbeddings），覆盖 api provider 初始化与 dim。"""
    import langchain_openai

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", _FakeClient)
    monkeypatch.setattr(config, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(config, "EMBEDDING_BASE_URL", "")
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 64)

    emb = APIEmbedder()
    assert emb.name == "api"
    assert emb.embed_documents(["a", "b"]) == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert emb.embed_query("q") == [0.1, 0.2, 0.3]
    assert emb.dim == 3



def test_check_embedding_health_ok(monkeypatch):
    import src.embeddings as emb

    monkeypatch.setattr(emb, "get_embedder", lambda force=False: _StubEmbedder())
    ok, msg = check_embedding_health()
    assert ok and "api" in msg


def test_check_embedding_health_fails(monkeypatch):
    import src.embeddings as emb

    def boom(force=False):
        raise RuntimeError("no key")

    monkeypatch.setattr(emb, "get_embedder", boom)
    ok, msg = check_embedding_health()
    assert ok is False and "RuntimeError" in msg
