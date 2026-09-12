"""RAG 主流程：解析 -> 清洗 -> 切分 -> Embedding -> 建库 -> 混合检索 -> 重排 -> 带引用生成。

本模块与业务无关，换一批文档即可作为企业知识库 / 智能客服复用。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .. import config
from ..cache import ResultCache, Timer, get_cache, get_cost_tracker
from ..embeddings import get_embedder
from ..logging_setup import get_logger
from ..models import Citation, RAGAnswer, RetrievedChunk
from ..safety import safe_output
from ..telemetry import span
from .bm25 import BM25Index
from .clean import clean_documents
from .fuse import dedup_by_text, rrf_fuse
from .loaders import load_paths
from .rerank import get_reranker
from .split import split_documents
from .types import Chunk
from .vectorstore import FaissStore

logger = get_logger(__name__)


class RetrievalConfig(BaseModel):
    """检索配置，可整体序列化用于评测与缓存键。"""

    use_bm25: bool = Field(default=config.USE_BM25)
    use_vector: bool = Field(default=config.USE_VECTOR)
    fusion: str = Field(default=config.FUSION)  # rrf | vector_only | bm25_only
    top_k: int = Field(default=config.TOP_K)
    reranker: str = Field(default=config.RERANK_PROVIDER)  # none | llm | api
    rerank_top_n: int = Field(default=config.RERANK_TOP_N)
    chunk_size: int = Field(default=config.CHUNK_SIZE)
    chunk_overlap: int = Field(default=config.CHUNK_OVERLAP)


class IngestResult(BaseModel):
    files: int = 0
    documents: int = 0
    chunks: int = 0
    dim: int = 0
    elapsed_ms: int = 0


class RAGPipeline:
    """单个知识库的检索增强生成流水线。"""

    ANSWER_PROMPT = (
        "你是一个严谨的知识库问答助手。只能依据下面提供的【资料片段】回答问题。\n"
        "规则：\n"
        "1. 回答要简洁、分点，必要时引用片段编号，如 [1]、[2]；\n"
        "2. 若资料不足以回答，直接说明「资料中未提及」，严禁编造；\n"
        "3. 使用与问题相同的语言回答。\n\n"
        "【资料片段】\n{context}\n\n【用户问题】\n{query}\n"
    )

    def __init__(self, name: str, index_dir: Path | None = None) -> None:
        self.name = name
        self.index_dir = Path(index_dir or config.INDEX_DIR / name)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[Chunk] = []
        self.store: FaissStore | None = None
        self.bm25: BM25Index | None = None
        self.dim: int = 0
        # 知识库内容指纹：纳入召回缓存键，保证重建库后旧缓存自动失效
        self._fingerprint: str = ""
        self._load_if_exists()

    # ------------------------------------------------------------ 建库
    def ingest(
        self,
        paths: Sequence[str | Path],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestResult:
        """解析并建库（覆盖式）。"""
        with span(
            "rag.ingest",
            {
                "kb.name": self.name,
                "kb.files": len(set(str(p) for p in paths)),
            },
        ) as sp:
            with Timer() as t:
                raw_docs = load_paths(paths)
                docs = clean_documents(raw_docs)
                if not docs:
                    return IngestResult(elapsed_ms=t.ms if hasattr(t, "ms") else 0)

                chunks = split_documents(docs, chunk_size, chunk_overlap)
                if not chunks:
                    return IngestResult(
                        files=len(set(str(p) for p in paths)),
                        documents=len(docs),
                        elapsed_ms=t.ms if hasattr(t, "ms") else 0,
                    )

                embedder = get_embedder()
                texts = [c.text for c in chunks]
                vectors: list[list[float]] = []
                for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
                    batch = texts[i : i + config.EMBEDDING_BATCH_SIZE]
                    vectors.extend(embedder.embed_documents(batch))
                    logger.debug("Embedding 进度 %d/%d", min(i + len(batch), len(texts)), len(texts))

                self.chunks = chunks
                self.dim = len(vectors[0]) if vectors else 0
                self.store = FaissStore(self.dim)
                self.store.add(chunks, vectors)

                self.bm25 = BM25Index()
                self.bm25.build(chunks)

                self._fingerprint = self._compute_fingerprint()
                self.save()

            sp.set_attribute("kb.chunks", len(chunks))
            sp.set_attribute("kb.documents", len(docs))
            sp.set_attribute("kb.dim", self.dim)
            sp.set_attribute("kb.elapsed_ms", t.ms)

        return IngestResult(
            files=len(set(str(p) for p in paths)),
            documents=len(docs),
            chunks=len(chunks),
            dim=self.dim,
            elapsed_ms=t.ms,
        )

    def save(self) -> None:
        if self.store:
            self.store.save(self.index_dir)
        if self.bm25:
            self.bm25.save(self.index_dir)

    def _load_if_exists(self) -> None:
        store = FaissStore.load(self.index_dir)
        bm25 = BM25Index.load(self.index_dir)
        if store is not None:
            self.store = store
            self.chunks = store.chunks
            self.dim = store.dim
        if bm25 is not None:
            self.bm25 = bm25
            if not self.chunks:
                self.chunks = bm25.chunks
        self._fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """知识库内容指纹：由片段 id 集合派生，随重建库而变化，用于缓存失效。

        chunk_id 含全局自增序号与文本摘要，改动切分参数或换文档都会改变指纹；
        用增量 md5 避免拼接大字符串，仅在加载/建库后计算一次。空库返回空串。
        """
        if not self.chunks:
            return ""
        h = hashlib.md5()
        h.update(str(len(self.chunks)).encode("utf-8"))
        for c in self.chunks:
            h.update(b"\x00")
            h.update(c.chunk_id.encode("utf-8"))
        return h.hexdigest()[:16]

    @property
    def ready(self) -> bool:
        return self.store is not None and len(self.chunks) > 0

    def stats(self) -> dict:
        return {
            "name": self.name,
            "chunks": len(self.chunks),
            "dim": self.dim,
            "has_vector": self.store is not None,
            "has_bm25": self.bm25 is not None,
            "sources": len({c.source for c in self.chunks}),
        }

    # ------------------------------------------------------------ 检索
    def _candidates(
        self,
        query: str,
        cfg: RetrievalConfig,
        sp: Any | None = None,
        use_cache: bool = True,
    ) -> list[RetrievedChunk]:
        """混合检索 + 融合，返回候选片段（未重排）。

        命中召回缓存时直接返回缓存候选，跳过重复的 query 向量化 + BM25 + RRF。
        缓存键含知识库指纹与召回参数、**不含 reranker**，使「仅重排方式不同」的
        多组实验复用同一召回；`use_cache=False` 时完全不读写缓存（评测测冷启动用）。
        """
        if not self.ready:
            return []

        cache = get_cache() if (use_cache and config.ENABLE_RETRIEVAL_CACHE) else None
        cache_key = None
        if cache is not None and cache.enabled:
            cache_key = ResultCache.make_key(
                "retrieval",
                self.name,
                self._fingerprint,
                query,
                cfg.use_bm25,
                cfg.use_vector,
                cfg.fusion,
                cfg.top_k,
                cfg.rerank_top_n,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                if sp is not None:
                    sp.set_attribute("rag.retrieval_cache_hit", True)
                return cached

        vector_hits: list = []
        bm25_hits: list = []

        if cfg.use_vector and self.store is not None:
            try:
                qv = get_embedder().embed_query(query)
                vector_hits = self.store.search(qv, top_k=max(cfg.rerank_top_n, cfg.top_k) * 2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("向量检索失败，跳过：%s", exc)

        if cfg.use_bm25 and self.bm25 is not None:
            try:
                bm25_hits = self.bm25.search(query, top_k=max(cfg.rerank_top_n, cfg.top_k) * 2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("BM25 检索失败，跳过：%s", exc)

        if cfg.fusion == "vector_only":
            fused = [(c, s, {"vector_rank": i + 1, "bm25_rank": None}) for i, (c, s) in enumerate(vector_hits)]
        elif cfg.fusion == "bm25_only":
            fused = [(c, s, {"vector_rank": None, "bm25_rank": i + 1}) for i, (c, s) in enumerate(bm25_hits)]
        else:
            fused = rrf_fuse(
                [vector_hits, bm25_hits],
                top_k=max(cfg.rerank_top_n, cfg.top_k) * 2,
            )

        fused = dedup_by_text(fused)
        candidates = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                page=c.page,
                score=float(score),
                vector_rank=info.get("vector_rank"),
                bm25_rank=info.get("bm25_rank"),
            )
            for c, score, info in fused
        ]
        # 仅缓存非空召回，避免建库前 / 空结果污染缓存
        if cache is not None and cache_key is not None and candidates:
            cache.set(cache_key, candidates, expire=config.RETRIEVAL_CACHE_TTL)
        return candidates

    def retrieve(
        self,
        query: str,
        cfg: RetrievalConfig | None = None,
        use_cache: bool = True,
    ) -> list[RetrievedChunk]:
        """检索 + 重排，返回最终用于生成的片段。

        `use_cache=False` 时跳过召回缓存，供评测测量「冷启动」检索延迟。
        """
        cfg = cfg or RetrievalConfig()
        with span(
            "rag.retrieve",
            {
                "kb.name": self.name,
                "rag.reranker": cfg.reranker,
                "rag.top_k": cfg.top_k,
                "rag.fusion": cfg.fusion,
                "rag.use_vector": cfg.use_vector,
                "rag.use_bm25": cfg.use_bm25,
                "rag.query_chars": len(query),
            },
        ) as sp:
            candidates = self._candidates(query, cfg, sp=sp, use_cache=use_cache)
            sp.set_attribute("rag.candidates", len(candidates))
            if not candidates:
                return []

            if (cfg.reranker or "none").lower() in {"none", "off", "false"}:
                final = candidates[: cfg.top_k]
            else:
                chunk_map = {c.chunk_id: c for c in self.chunks}
                objs = [chunk_map.get(c.chunk_id) for c in candidates[: cfg.rerank_top_n]]
                objs = [o for o in objs if o is not None]
                reranker = get_reranker(cfg.reranker)
                reranked = reranker.rerank(query, objs, top_n=cfg.top_k)
                final = []
                for chunk, score in reranked:
                    rc = RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        source=chunk.source,
                        page=chunk.page,
                        score=float(score),
                        rerank_score=float(score),
                    )
                    final.append(rc)
            sp.set_attribute("rag.results", len(final))
            return final

    # ------------------------------------------------------------ 生成
    def answer(self, query: str, cfg: RetrievalConfig | None = None) -> RAGAnswer:
        """检索 + 带引用生成（相同 知识库+query+检索配置 走结果缓存，省 token 且省检索）。"""
        cfg = cfg or RetrievalConfig()
        with span("rag.answer", {"kb.name": self.name, "rag.top_k": cfg.top_k, "rag.reranker": cfg.reranker}) as sp:
            # 回答结果缓存前移到检索之前：命中时真正跳过重复检索（含 query 向量化）
            cache = get_cache()
            cache_key = None
            if cache.enabled:
                cache_key = ResultCache.make_key(self.name, self._fingerprint, query, cfg.model_dump())
                cached = cache.get(cache_key)
                if cached is not None:
                    sp.set_attribute("rag.cache_hit", True)
                    return cached

            with Timer() as t:
                chunks = self.retrieve(query, cfg)
            if not chunks:
                return RAGAnswer(
                    answer="知识库中未检索到相关内容。请先上传文档并建库，或换个问法。",
                    latency_ms=t.ms if hasattr(t, "ms") else 0,
                )

            context = "\n\n".join(
                f"[{i + 1}] 来源：{c.source}{' 第' + str(c.page) + '页' if c.page else ''}\n{c.text}"
                for i, c in enumerate(chunks)
            )
            prompt = self.ANSWER_PROMPT.format(context=context, query=query)

            from langchain_core.messages import HumanMessage

            from ..llm import llm_invoke

            raw = llm_invoke([HumanMessage(content=prompt)])
            text = safe_output(raw)
            get_cost_tracker().record(prompt, text, latency_ms=t.ms)

            citations = [
                Citation(index=i + 1, source=c.source, page=c.page, snippet=c.text[:200])
                for i, c in enumerate(chunks)
            ]

            sp.set_attribute("rag.chunks", len(chunks))
            sp.set_attribute("rag.answer_chars", len(text or ""))
            sp.set_attribute("rag.latency_ms", t.ms)

        answer = RAGAnswer(
            answer=text,
            citations=citations,
            chunks=chunks,
            latency_ms=t.ms,
            tokens=getattr(get_cost_tracker(), "completion_tokens", 0),
        )
        if cache_key is not None:
            cache.set(cache_key, answer, expire=3600)
        return answer

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None
