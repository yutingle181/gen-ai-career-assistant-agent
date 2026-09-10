"""BM25 稀疏检索。

与向量检索互补：BM25 擅长岗位名、技术栈、专有名词等关键词精确匹配，
向量检索擅长语义召回，二者用 RRF 融合（见 fuse.py）。
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from ..logging_setup import get_logger
from .types import Chunk

logger = get_logger(__name__)

# 中文逐字切分 + 英文/数字按词切分（不引入 jieba，保持零额外依赖）
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+")


def tokenize(text: str) -> list[str]:
    """轻量分词：中文按字，英文数字按词。"""
    return _TOKEN_PATTERN.findall(text or "")


class BM25Index:
    """BM25 索引封装。"""

    def __init__(self) -> None:
        self._bm25 = None
        self.chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self._tokens = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None
        logger.info("BM25 索引构建完成 | %d 片段", len(self.chunks))

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        if self._bm25 is None or not self.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[i], float(score)) for i, score in ranked if score > 0]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "bm25.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "tokens": self._tokens}, f)

    @classmethod
    def load(cls, directory: str | Path) -> BM25Index | None:
        path = Path(directory) / "bm25.pkl"
        if not path.exists():
            return None
        try:
            from rank_bm25 import BM25Okapi

            with open(path, "rb") as f:
                data = pickle.load(f)
            idx = cls()
            idx.chunks = data["chunks"]
            idx._tokens = data["tokens"]
            idx._bm25 = BM25Okapi(idx._tokens) if idx._tokens else None
            return idx
        except Exception as exc:  # noqa: BLE001
            logger.error("BM25 索引加载失败：%s", exc)
            return None

    def __len__(self) -> int:
        return len(self.chunks)
