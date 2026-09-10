"""FAISS 向量库封装：建库、检索、持久化。

使用 IndexFlatIP + L2 归一化向量（等价于余弦相似度），
纯 CPU 运行，索引与元数据一起落盘，进程重启无需重建。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..logging_setup import get_logger
from .types import Chunk

logger = get_logger(__name__)


class FaissStore:
    """轻量 FAISS 封装。"""

    def __init__(self, dim: int) -> None:
        import faiss  # 延迟导入，模块加载更快

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    # ---------------------------------------------------------- 写入
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.shape[1] != self.dim:
            raise ValueError(f"向量维度 {matrix.shape[1]} 与索引维度 {self.dim} 不一致")
        # 归一化后内积 == 余弦相似度
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self.index.add(matrix)
        self.chunks.extend(chunks)

    def search(self, vector: list[float], top_k: int = 10) -> list[tuple[Chunk, float]]:
        """检索，返回 [(片段, 相似度)]，按分数降序。"""
        if not self.chunks:
            return []
        q = np.asarray([vector], dtype="float32")
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        k = min(top_k, len(self.chunks))
        scores, ids = self.index.search(q, k)
        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(ids[0], scores[0], strict=False):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[int(idx)], float(score)))
        return results

    # ---------------------------------------------------------- 持久化
    def save(self, directory: str | Path) -> None:
        import faiss

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        with open(d / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(d / "meta.json", "w", encoding="utf-8") as f:
            import json

            json.dump({"dim": self.dim, "count": len(self.chunks)}, f, ensure_ascii=False)
        logger.info("向量索引已保存 | %s | %d 片段", d, len(self.chunks))

    @classmethod
    def load(cls, directory: str | Path) -> FaissStore | None:
        import faiss

        d = Path(directory)
        index_path = d / "index.faiss"
        chunks_path = d / "chunks.pkl"
        if not (index_path.exists() and chunks_path.exists()):
            return None
        try:
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks: list[Chunk] = pickle.load(f)
            store = cls(index.d)
            store.index = index
            store.chunks = chunks
            logger.info("向量索引已加载 | %s | %d 片段", d, len(chunks))
            return store
        except Exception as exc:  # noqa: BLE001
            logger.error("向量索引加载失败：%s", exc)
            return None

    # ---------------------------------------------------------- 信息
    def __len__(self) -> int:
        return len(self.chunks)
