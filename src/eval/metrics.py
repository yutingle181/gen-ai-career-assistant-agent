"""评测指标。

口径固定，保证报告可复现：
- Recall@K：golden 片段是否出现在前 K 条召回中（本评测集每条仅 1 个 golden）；
- MRR：golden 片段首次命中排名的倒数，未命中记 0；
- Hit Rate@K：前 K 条至少命中一条的样本占比；
- 幻觉率：LLM-as-judge 把回答拆成断言，统计不可由召回片段支撑的断言占比。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ..logging_setup import get_logger
from ..models import RetrievedChunk
from ..prompts.rag import HALLUCINATION_JUDGE_PROMPT

logger = get_logger(__name__)


def recall_at_k(retrieved_ids: Sequence[str], golden_ids: Sequence[str], k: int = 5) -> float:
    if not golden_ids:
        return 0.0
    top = set(list(retrieved_ids)[:k])
    return 1.0 if top & set(golden_ids) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], golden_ids: Sequence[str]) -> float:
    golden = set(golden_ids)
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in golden:
            return 1.0 / i
    return 0.0


def hit_rate(samples: Sequence[tuple[Sequence[str], Sequence[str]]], k: int = 5) -> float:
    if not samples:
        return 0.0
    hits = sum(recall_at_k(r, g, k) for r, g in samples)
    return hits / len(samples)


def evaluate_retrieval(
    samples: Sequence[tuple[Sequence[str], Sequence[str]]], k: int = 5
) -> tuple[float, float, float]:
    """返回 (recall@k, mrr, hit_rate@k)。"""
    if not samples:
        return 0.0, 0.0, 0.0
    recall = sum(recall_at_k(r, g, k) for r, g in samples) / len(samples)
    mrr = sum(reciprocal_rank(r, g) for r, g in samples) / len(samples)
    hr = hit_rate(samples, k)
    return recall, mrr, hr


# ---------------------------------------------------------------- 幻觉判定
def judge_hallucination(answer: str, chunks: Sequence[RetrievedChunk], model: str = "") -> tuple[float, int]:
    """用 LLM 判定回答的事实一致性，返回 (幻觉率, 断言数)。"""
    if not answer or not chunks:
        return 0.0, 0
    from langchain_core.messages import HumanMessage

    from ..llm import llm_invoke

    context = "\n\n".join(f"[{i + 1}] {c.text[:800]}" for i, c in enumerate(chunks))
    prompt = HALLUCINATION_JUDGE_PROMPT.format(context=context, answer=answer[:2000])
    try:
        raw = llm_invoke([HumanMessage(content=prompt)], temperature=0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("幻觉判定失败：%s", exc)
        return 0.0, 0

    claims = _parse_claims(raw)
    if not claims:
        return 0.0, 0
    unsupported = sum(1 for c in claims if not c.get("supported", True))
    return unsupported / len(claims), len(claims)


def _parse_claims(raw: str) -> list[dict]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except Exception:  # noqa: BLE001
        return []
    claims = data.get("claims", [])
    return [c for c in claims if isinstance(c, dict)]
