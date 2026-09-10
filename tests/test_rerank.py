"""重排层单元测试（离线、确定性）。

覆盖 none/llm/api 三种重排器、失败回退、打分解析、工厂分发。
通过 monkeypatch `src.llm.llm_invoke` 与 `requests.post` 避免真实网络。
"""

from __future__ import annotations

from src.rag.rerank import (
    APIReranker,
    LLMReranker,
    NoneReranker,
    get_reranker,
)
from src.rag.types import Chunk


def _chunks():
    return [
        Chunk(chunk_id="a", text="混合检索结合 BM25 与向量", source="a.md"),
        Chunk(chunk_id="b", text="RRF 融合无需权重", source="b.md"),
        Chunk(chunk_id="c", text="重排提升准确率", source="c.md"),
    ]


def test_none_reranker_keeps_order():
    out = NoneReranker().rerank("q", _chunks(), top_n=2)
    assert [c.chunk_id for c, _ in out] == ["a", "b"]


def test_llm_reranker_empty():
    assert LLMReranker().rerank("q", [], top_n=3) == []


def test_llm_reranker_parse_variants():
    r = LLMReranker()
    assert r._parse("", 3) == {}
    assert r._parse("没有 json", 3) == {}
    assert r._parse('{"scores": "bad"}', 3) == {}
    assert r._parse('```json\n{"scores":[{"index":1,"score":9}]}\n```', 3) == {0: 9.0}


def test_llm_reranker_success(monkeypatch):
    import src.llm as llm

    monkeypatch.setattr(
        llm,
        "llm_invoke",
        lambda msgs, temperature=None: '{"scores":[{"index":2,"score":8},{"index":1,"score":5}]}',
    )
    out = LLMReranker().rerank("q", _chunks(), top_n=2)
    assert [c.chunk_id for c, _ in out] == ["b", "a"]


def test_llm_reranker_fallback_on_error(monkeypatch):
    import src.llm as llm

    def boom(msgs, temperature=None):
        raise RuntimeError("llm down")

    monkeypatch.setattr(llm, "llm_invoke", boom)
    out = LLMReranker().rerank("q", _chunks(), top_n=2)
    assert [c.chunk_id for c, _ in out] == ["a", "b"]  # 失败回退原顺序


def test_api_reranker_empty():
    assert APIReranker().rerank("q", [], top_n=3) == []


def test_api_reranker_success(monkeypatch):
    import requests

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.3}]}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    out = APIReranker().rerank("q", _chunks(), top_n=2)
    # 返回的 index 2 -> candidates[2]="c"、index 0 -> candidates[0]="a"
    assert [c.chunk_id for c, _ in out] == ["c", "a"]


def test_api_reranker_fallback_on_error(monkeypatch):
    import requests

    def boom(*a, **k):
        raise RuntimeError("net down")

    monkeypatch.setattr(requests, "post", boom)
    out = APIReranker().rerank("q", _chunks(), top_n=2)
    assert [c.chunk_id for c, _ in out] == ["a", "b"]  # 网络失败回退原顺序


def test_get_reranker_dispatch():
    assert get_reranker("none").name == "none"
    assert get_reranker("off").name == "none"
    assert get_reranker("false").name == "none"
    assert get_reranker("api").name == "api"
    assert get_reranker(None).name == "llm"  # 默认实现
