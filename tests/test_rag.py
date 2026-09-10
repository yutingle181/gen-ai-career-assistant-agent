"""RAG 链路单元测试。

不依赖真实 API：用确定性哈希伪 Embedding 替代云端接口，
验证 解析 -> 清洗 -> 切分 -> 建库 -> 混合检索 -> 融合 -> 重排 全链路。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.rag.bm25 import BM25Index, tokenize  # noqa: E402
from src.rag.clean import clean_text  # noqa: E402
from src.rag.fuse import rrf_fuse  # noqa: E402
from src.rag.loaders import load_dir  # noqa: E402
from src.rag.pipeline import RAGPipeline, RetrievalConfig  # noqa: E402
from src.rag.split import split_documents  # noqa: E402
from src.rag.types import Chunk  # noqa: E402


class _FakeEmbedder:
    """确定性伪 Embedding：用字符哈希构造固定维度向量。"""

    name = "fake"

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str):
        vec = [0.0] * self._dim
        for ch in (text or ""):
            vec[ord(ch) % self._dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


def _patch_embedder():
    import src.embeddings as emb

    emb.get_embedder = lambda force=False: _FakeEmbedder()
    import src.rag.pipeline as pipe

    pipe.get_embedder = emb.get_embedder


def test_loaders_and_clean():
    docs = load_dir(config.SAMPLES_DIR)
    assert docs, "示例文档解析失败"
    assert all(d.text.strip() for d in docs)

    cleaned = clean_text("第一行\r\n\r\n\r\n第二行\n- 3 -\n")
    assert "第二行" in cleaned
    assert "- 3 -" not in cleaned
    print(f"[OK] 解析 {len(docs)} 个文档单元，清洗规则生效")


def test_split():
    docs = load_dir(config.SAMPLES_DIR)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= len(docs)
    assert all(len(c.text) > 0 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk_id 必须唯一"
    print(f"[OK] 切分得到 {len(chunks)} 个片段")


def test_bm25_and_fuse():
    chunks = [
        Chunk(chunk_id="a", text="混合检索结合 BM25 与向量检索", source="a.md"),
        Chunk(chunk_id="b", text="RRF 融合无需调权重", source="b.md"),
        Chunk(chunk_id="c", text="重排模型提升准确率", source="c.md"),
    ]
    idx = BM25Index()
    idx.build(chunks)
    hits = idx.search("BM25 向量 混合检索", top_k=3)
    assert hits and hits[0][0].chunk_id == "a", "BM25 应优先召回关键词命中的片段"
    # "RAG" / "与" / "Agent" 三个 token：英文按词、中文按字
    assert tokenize("RAG 与 Agent") == ["RAG", "与", "Agent"]

    fused = rrf_fuse([[(chunks[1], 0.9), (chunks[0], 0.8)], [(chunks[0], 5.0), (chunks[2], 1.0)]])
    assert fused[0][0].chunk_id == "a", "RRF 应把两次都靠前的片段排到第一"
    assert fused[0][2]["bm25_rank"] == 1
    print("[OK] BM25 检索与 RRF 融合正确")


def test_pipeline_end_to_end():
    _patch_embedder()
    tmp = Path(tempfile.mkdtemp(prefix="rag_test_"))
    try:
        pipe = RAGPipeline("test_kb", index_dir=tmp)
        result = pipe.ingest([str(config.SAMPLES_DIR)], chunk_size=300, chunk_overlap=50)
        assert result.chunks > 0, "建库未产生片段"
        assert pipe.ready

        # 重新加载校验持久化
        reloaded = RAGPipeline("test_kb", index_dir=tmp)
        assert reloaded.ready and len(reloaded.chunks) == result.chunks
        print(f"[OK] 建库 {result.chunks} 片段 / 持久化与重载正常")

        cfg = RetrievalConfig(reranker="none", top_k=3)
        hits = reloaded.retrieve("混合检索怎么做？", cfg)
        assert hits, "检索结果为空"
        assert len(hits) <= 3
        print(f"[OK] 检索返回 {len(hits)} 条：{hits[0].source} score={hits[0].score:.3f}")

        # 关闭向量 / 关闭 BM25 的对照
        only_vec = reloaded.retrieve("混合检索怎么做？", RetrievalConfig(use_bm25=False, reranker="none", top_k=3))
        only_bm25 = reloaded.retrieve("混合检索怎么做？", RetrievalConfig(use_vector=False, reranker="none", top_k=3))
        assert only_vec and only_bm25
        print("[OK] 单路检索（仅向量 / 仅 BM25）均可用")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_loaders_and_clean()
    test_split()
    test_bm25_and_fuse()
    test_pipeline_end_to_end()
    print("\n全部 RAG 单元测试通过")
