"""召回结果缓存（9.2.B）单元测试。

不依赖真实 API：用确定性伪 Embedding（带调用计数）替换云端接口，
并注入隔离的 ResultCache（临时目录），验证：
- 相同 query + 相同召回参数：命中召回缓存，跳过 query 向量化；
- 缓存键不含 reranker：仅「重排方式」不同仍复用同一召回；
- 不同 query / ENABLE_RETRIEVAL_CACHE=False / use_cache=False：不命中，正常重算；
- 重建库后知识库指纹变化：旧召回缓存自然失效；
- answer 回答缓存前移到检索之前：命中时连检索一起跳过。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.cache import ResultCache  # noqa: E402
from src.rag.pipeline import RAGPipeline, RetrievalConfig  # noqa: E402

_DOC = (
    "混合检索结合 BM25 与向量检索两条召回路径。\n"
    "BM25 擅长关键词精确匹配，向量检索擅长语义召回。\n"
    "RRF（Reciprocal Rank Fusion）用于融合两路结果，无需调权重。\n"
    "重排模型可对候选片段做更精细的相关性排序，提升准确率。\n"
    "知识库问答需要引用来源，避免编造。\n"
)

_DOC2 = (
    "切分参数会影响片段边界与 chunk_id，从而改变知识库指纹。\n"
    "嵌入向量用确定性哈希构造，便于离线测试。\n"
    "缓存命中时应跳过 query 向量化，直接复用召回候选。\n"
)


class _CountingEmbedder:
    """确定性伪 Embedding，统计 embed_query 调用次数。"""

    name = "fake-counting"

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.query_calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str):
        self.query_calls += 1
        return self._embed(text)

    def _embed(self, text: str):
        vec = [0.0] * self._dim
        for ch in (text or ""):
            vec[ord(ch) % self._dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


class _Harness:
    """临时建库 + 注入隔离缓存的测试脚手架。"""

    def __init__(self, kb_name: str = "cache_kb") -> None:
        self.kb_name = kb_name
        self.root = Path(tempfile.mkdtemp(prefix="rag_cache_test_"))
        self.index_dir = self.root / "index"
        self.cache = ResultCache(str(self.root / "cache"), enabled=True)
        self.embedder = _CountingEmbedder()
        self.pipe: RAGPipeline | None = None
        self._orig: dict = {}

    def __enter__(self) -> _Harness:
        import src.rag.pipeline as pipe_mod

        self._orig["get_embedder"] = pipe_mod.get_embedder
        self._orig["get_cache"] = pipe_mod.get_cache
        pipe_mod.get_embedder = lambda force=False: self.embedder
        pipe_mod.get_cache = lambda: self.cache
        return self

    def __exit__(self, *exc) -> None:
        import src.rag.pipeline as pipe_mod

        pipe_mod.get_embedder = self._orig["get_embedder"]
        pipe_mod.get_cache = self._orig["get_cache"]
        shutil.rmtree(self.root, ignore_errors=True)

    def build(self, text: str, name: str | None = None, **ingest_kwargs) -> RAGPipeline:
        name = name or self.kb_name
        docs = self.root / f"docs_{name}"
        docs.mkdir(parents=True, exist_ok=True)
        (docs / f"{name}.md").write_text(text, encoding="utf-8")
        self.pipe = RAGPipeline(name, index_dir=self.index_dir)
        self.pipe.ingest([str(docs)], chunk_size=200, chunk_overlap=40, **ingest_kwargs)
        self.embedder.query_calls = 0  # 建库向量化不计入检索计数
        return self.pipe

    def reingest(self, text: str, name: str | None = None, **ingest_kwargs) -> None:
        name = name or self.kb_name
        docs = self.root / f"docs_{name}"
        (docs / f"{name}.md").write_text(text, encoding="utf-8")
        assert self.pipe is not None
        self.pipe.ingest([str(docs)], chunk_size=150, chunk_overlap=20, **ingest_kwargs)


def _cfg(**overrides) -> RetrievalConfig:
    base = dict(use_bm25=True, use_vector=True, fusion="rrf", reranker="none", top_k=3, rerank_top_n=10)
    base.update(overrides)
    return RetrievalConfig(**base)


def test_hit_same_query():
    with _Harness() as h:
        pipe = h.build(_DOC)
        cfg = _cfg()
        first = pipe.retrieve("混合检索怎么做？", cfg)
        n1 = h.embedder.query_calls
        assert first, "首次检索应有结果"
        assert n1 >= 1, "首次检索应触发 query 向量化"

        second = pipe.retrieve("混合检索怎么做？", cfg)
        assert h.embedder.query_calls == n1, "二次检索应命中召回缓存，不再向量化"
        assert [c.chunk_id for c in second] == [c.chunk_id for c in first], "缓存候选应与首次一致"
        print(f"[OK] 相同 query 命中召回缓存（embed 调用 {n1} 次，返回 {len(second)} 条）")


def test_key_excludes_reranker():
    with _Harness() as h:
        pipe = h.build(_DOC)
        a = pipe._candidates("混合检索怎么做？", _cfg(reranker="none"))
        n1 = h.embedder.query_calls
        assert n1 == 1, "首次召回应向量化一次"

        # 仅 reranker 不同：缓存键不含 reranker，应复用同一召回
        b = pipe._candidates("混合检索怎么做？", _cfg(reranker="llm"))
        assert h.embedder.query_calls == n1, "仅 reranker 不同应复用召回，不应再向量化"
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
        print("[OK] 缓存键不含 reranker：仅重排方式不同复用同一召回")


def test_different_query_recomputes():
    with _Harness() as h:
        pipe = h.build(_DOC)
        cfg = _cfg()
        pipe.retrieve("混合检索怎么做？", cfg)
        n1 = h.embedder.query_calls
        pipe.retrieve("完全不同的另一个问题？", cfg)
        assert h.embedder.query_calls == n1 + 1, "不同 query 应重新向量化"
        print("[OK] 不同 query 未命中缓存，正常重算")


def test_use_cache_false_bypasses_and_does_not_write():
    with _Harness() as h:
        pipe = h.build(_DOC)
        cfg = _cfg()
        pipe.retrieve("混合检索怎么做？", cfg, use_cache=False)
        pipe.retrieve("混合检索怎么做？", cfg, use_cache=False)
        assert h.embedder.query_calls == 2, "use_cache=False 应每次都重算（评测测冷启动）"

        # 上面的旁路调用不应写入缓存：默认调用仍应未命中
        pipe.retrieve("混合检索怎么做？", cfg)
        assert h.embedder.query_calls == 3, "use_cache=False 不应写缓存"
        print("[OK] use_cache=False 旁路召回缓存且不写入")


def test_switch_off_recomputes():
    with _Harness() as h:
        pipe = h.build(_DOC)
        cfg = _cfg()
        orig = config.ENABLE_RETRIEVAL_CACHE
        config.ENABLE_RETRIEVAL_CACHE = False
        try:
            pipe.retrieve("混合检索怎么做？", cfg)
            pipe.retrieve("混合检索怎么做？", cfg)
        finally:
            config.ENABLE_RETRIEVAL_CACHE = orig
        assert h.embedder.query_calls == 2, "开关关闭时应完全不走缓存"
        print("[OK] ENABLE_RETRIEVAL_CACHE=False 零缓存")


def test_fingerprint_invalidates_after_reingest():
    with _Harness() as h:
        pipe = h.build(_DOC)
        cfg = _cfg()
        pipe.retrieve("混合检索怎么做？", cfg)
        n1 = h.embedder.query_calls
        fp1 = pipe._fingerprint

        # 重建库：内容与切分均变化 -> chunk_id 与指纹变化 -> 旧缓存不应再命中
        h.reingest(_DOC2)
        assert pipe._fingerprint != fp1, "重建库后指纹应变化"
        pipe.retrieve("混合检索怎么做？", cfg)
        assert h.embedder.query_calls == n1 + 1, "指纹变化后旧召回缓存应失效"
        print("[OK] 重建库指纹变化使旧召回缓存失效")


def test_answer_cache_skips_retrieval():
    with _Harness() as h:
        pipe = h.build(_DOC)
        import src.llm as llm_mod

        calls = {"n": 0}

        def _fake_invoke(messages, **kwargs):
            calls["n"] += 1
            return "这是回答，引用片段 [1]。"

        orig_invoke = llm_mod.llm_invoke
        llm_mod.llm_invoke = _fake_invoke
        try:
            cfg = _cfg()
            a1 = pipe.answer("混合检索怎么做？", cfg)
            n_embed = h.embedder.query_calls
            assert calls["n"] == 1, "首次问答应调用一次模型"

            a2 = pipe.answer("混合检索怎么做？", cfg)
            assert calls["n"] == 1, "二次问答应命中回答缓存，不再打模型"
            assert h.embedder.query_calls == n_embed, "命中回答缓存时应连检索一起跳过"
            assert a2.answer == a1.answer
        finally:
            llm_mod.llm_invoke = orig_invoke
        print("[OK] answer 回答缓存前移：命中时跳过检索与生成")


if __name__ == "__main__":
    test_hit_same_query()
    test_key_excludes_reranker()
    test_different_query_recomputes()
    test_use_cache_false_bypasses_and_does_not_write()
    test_switch_off_recomputes()
    test_fingerprint_invalidates_after_reingest()
    test_answer_cache_skips_retrieval()
    print("\n全部召回缓存单元测试通过")
