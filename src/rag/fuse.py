"""检索结果融合：RRF（Reciprocal Rank Fusion）。

RRF 公式：score(d) = Σ  1 / (k + rank_i(d))，k 默认 60。
优点：无需归一化不同检索器的分值，也无需调权重，稳健且可解释。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..logging_setup import get_logger
from .types import Chunk

logger = get_logger(__name__)

RRF_K = 60


def rrf_fuse(
    ranked_lists: Sequence[Sequence[tuple[Chunk, float]]],
    k: int = RRF_K,
    top_k: int = 10,
) -> list[tuple[Chunk, float, dict[str, int | None]]]:
    """融合多个有序检索结果。

    返回 [(片段, 融合分, {"vector_rank": int|None, "bm25_rank": int|None})]。
    """
    fused: dict[str, float] = {}
    ranks: dict[str, dict[str, int | None]] = {}
    chunk_map: dict[str, Chunk] = {}

    keys = ["vector_rank", "bm25_rank", "rank_3", "rank_4"]
    for list_idx, ranked in enumerate(ranked_lists):
        key = keys[list_idx] if list_idx < len(keys) else f"rank_{list_idx + 1}"
        for pos, (chunk, _score) in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + pos)
            info = ranks.setdefault(cid, {"vector_rank": None, "bm25_rank": None})
            if key in info:
                info[key] = pos
            chunk_map[cid] = chunk

    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk_map[cid], score, ranks[cid]) for cid, score in ordered]


def dedup_by_text(
    results: Sequence[tuple[Chunk, float, dict[str, int | None]]],
    max_chars: int = 1500,
) -> list[tuple[Chunk, float, dict[str, int | None]]]:
    """去重并限制送入重排/生成的总字符数。"""
    seen = set()
    out = []
    total = 0
    for chunk, score, info in results:
        sig = chunk.text.strip()[:80]
        if sig in seen:
            continue
        seen.add(sig)
        out.append((chunk, score, info))
        total += len(chunk.text)
        if total >= max_chars:
            break
    return out
