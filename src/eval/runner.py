"""评测执行器：多配置 A-B 跑批。

默认实验组（直接对应简历/面试可讲的优化点）：
1. 仅向量检索
2. 仅 BM25
3. 混合检索（RRF）
4. 混合检索 + LLM 重排
5. 混合检索 + API 重排（可选，失败自动跳过）

性能设计（针对「评测跑得慢」的优化）：
- 检索与幻觉判定都按样本并发，线程数由 `config.EVAL_MAX_WORKERS` 控制。
- 幻觉率默认**只判定一次**并复用给所有实验组（`judge_once=True`）：
  它衡量的是「知识库内容本身能否支撑结论」，与重排/排序方式无关，
  逐组重复判定只会成倍放大 LLM 调用而不改变结论。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from .. import config
from ..logging_setup import get_logger
from ..models import EvalRecord, MetricResult
from ..rag.pipeline import RAGPipeline, RetrievalConfig
from .metrics import evaluate_retrieval, judge_hallucination

logger = get_logger(__name__)


def default_experiments(include_api_rerank: bool = False) -> list[tuple[str, RetrievalConfig]]:
    """返回默认实验组 (名称, 配置)。"""
    base = dict(top_k=5, rerank_top_n=20)
    experiments = [
        ("仅向量检索", RetrievalConfig(use_bm25=False, use_vector=True, fusion="vector_only", reranker="none", **base)),
        ("仅 BM25", RetrievalConfig(use_bm25=True, use_vector=False, fusion="bm25_only", reranker="none", **base)),
        ("混合检索(RRF)", RetrievalConfig(use_bm25=True, use_vector=True, fusion="rrf", reranker="none", **base)),
        ("混合检索 + LLM重排", RetrievalConfig(use_bm25=True, use_vector=True, fusion="rrf", reranker="llm", **base)),
    ]
    if include_api_rerank:
        experiments.append(
            (
                "混合检索 + API重排",
                RetrievalConfig(use_bm25=True, use_vector=True, fusion="rrf", reranker="api", **base),
            )
        )
    return experiments


# ---------------------------------------------------------------- 并发工具
def _resolve_workers(max_workers: int | None, total: int) -> int:
    """确定实际并发数：不超过样本数，且至少为 1。"""
    workers = max_workers if max_workers and max_workers > 0 else config.EVAL_MAX_WORKERS
    return max(1, min(workers, max(1, total)))


def _map_concurrent(func, items: Sequence, workers: int) -> list:
    """按 workers 并发执行并保持输入顺序；workers=1 时退化为串行（便于调试与测试）。"""
    items = list(items)
    if not items:
        return []
    if workers <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, items))


# ---------------------------------------------------------------- 单样本任务
def _eval_one(
    pipeline: RAGPipeline,
    rec: EvalRecord,
    cfg: RetrievalConfig,
) -> tuple[list, list[str], int] | None:
    """跑单条样本的检索，返回 (召回片段, golden_chunk_ids, 耗时ms)；失败返回 None。"""
    try:
        start = time.perf_counter()
        # 评测需测「冷启动」检索延迟：跳过召回缓存，避免基线预热后同配置组延迟被打平
        chunks = pipeline.retrieve(rec.question, cfg, use_cache=False)
        latency = int((time.perf_counter() - start) * 1000)
    except Exception as exc:  # noqa: BLE001
        logger.warning("检索失败，跳过样本：%s", exc)
        return None
    return chunks, list(rec.golden_chunk_ids), latency


def _judge_chunks(chunks: Sequence) -> tuple[float, int]:
    """对一组已召回片段做幻觉判定，返回 (幻觉率, 断言数)。"""
    if not chunks:
        return 0.0, 0
    # 用「召回片段拼接」作为回答代理，衡量的是资料本身能否支撑结论
    answer = " ".join(c.text[:300] for c in chunks[:3])
    try:
        return judge_hallucination(answer, chunks)
    except Exception as exc:  # noqa: BLE001
        logger.debug("幻觉判定跳过：%s", exc)
        return 0.0, 0


def _judge_one(
    pipeline: RAGPipeline,
    rec: EvalRecord,
    cfg: RetrievalConfig,
) -> tuple[float, int]:
    """检索 + 幻觉判定（用于一次性基线判定）。"""
    try:
        chunks = pipeline.retrieve(rec.question, cfg, use_cache=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("基线检索失败，跳过该样本判定：%s", exc)
        return 0.0, 0
    return _judge_chunks(chunks)


def _judge_rate(
    pipeline: RAGPipeline,
    records: Sequence[EvalRecord],
    cfg: RetrievalConfig,
    workers: int,
) -> tuple[float, int]:
    """在给定配置上算出整体幻觉率，返回 (率, 断言总数)。"""
    outs = _map_concurrent(lambda r: _judge_one(pipeline, r, cfg), records, workers)
    rates = [rate for rate, n in outs if n]
    return (sum(rates) / len(rates) if rates else 0.0), sum(n for _, n in outs)


def _baseline_config(experiments: Sequence[tuple[str, RetrievalConfig]]) -> RetrievalConfig:
    """挑选用于「一次性判定幻觉率」的基线配置：优先混合检索且不重排的那组。"""
    no_rerank = {"none", "off", "false"}

    def _is_none(cfg: RetrievalConfig) -> bool:
        return (cfg.reranker or "none").lower() in no_rerank

    for _name, cfg in experiments:
        if cfg.fusion == "rrf" and _is_none(cfg):
            return cfg
    for _name, cfg in experiments:
        if _is_none(cfg):
            return cfg
    return experiments[0][1]


# ---------------------------------------------------------------- 实验执行
def run_experiment(
    pipeline: RAGPipeline,
    records: Sequence[EvalRecord],
    cfg: RetrievalConfig,
    name: str,
    k: int = 5,
    with_hallucination: bool = True,
    max_workers: int | None = None,
    hallucination_rate: float | None = None,
) -> MetricResult:
    """跑一组实验，返回指标。

    - `hallucination_rate` 传入时直接复用，不再逐样本判定；
    - 为 None 且 `with_hallucination` 时，按本组检索结果并发判定。
    """
    records = list(records)
    workers = _resolve_workers(max_workers, len(records))

    outs = _map_concurrent(lambda r: _eval_one(pipeline, r, cfg), records, workers)
    samples: list[tuple[list, list]] = []
    latencies: list[int] = []
    chunk_lists: list = []
    for out in outs:
        if out is None:
            continue
        chunks, golden, latency = out
        samples.append(([c.chunk_id for c in chunks], golden))
        latencies.append(latency)
        chunk_lists.append(chunks)

    if hallucination_rate is None and with_hallucination and chunk_lists:
        judged = _map_concurrent(_judge_chunks, chunk_lists, workers)
        hall_rates = [rate for rate, n in judged if n]
        hallucination_rate = sum(hall_rates) / len(hall_rates) if hall_rates else 0.0

    recall, mrr, hr = evaluate_retrieval(samples, k=k)
    result = MetricResult(
        name=name,
        recall_at_k=round(recall, 4),
        mrr=round(mrr, 4),
        hit_rate=round(hr, 4),
        hallucination_rate=round(hallucination_rate or 0.0, 4),
        avg_latency_ms=round(sum(latencies) / len(latencies)) if latencies else 0,
        sample_count=len(samples),
        config=cfg.model_dump(),
    )
    logger.info(
        "实验完成 | %s | Recall@%d=%.3f MRR=%.3f 延迟=%dms 样本=%d",
        name, k, recall, mrr, result.avg_latency_ms, len(samples),
    )
    return result


def run_all(
    pipeline: RAGPipeline,
    records: Sequence[EvalRecord],
    experiments: list[tuple[str, RetrievalConfig]] | None = None,
    k: int = 5,
    with_hallucination: bool = True,
    max_workers: int | None = None,
    judge_once: bool = True,
) -> list[MetricResult]:
    """跑全部实验组。

    `judge_once=True`（默认）时，幻觉率只在基线配置上判定一次并由各组复用；
    设为 False 则逐组判定（更慢，用于确实需要区分各组幻觉率的场景）。
    """
    experiments = experiments or default_experiments()
    records = list(records)
    workers = _resolve_workers(max_workers, len(records))

    shared_rate: float | None = None
    if with_hallucination and judge_once and records:
        baseline_cfg = _baseline_config(experiments)
        shared_rate, claims = _judge_rate(pipeline, records, baseline_cfg, workers)
        logger.info("幻觉率已在基线上判定一次：%.4f（断言 %d 条）", shared_rate, claims)

    results: list[MetricResult] = []
    for name, cfg in experiments:
        results.append(
            run_experiment(
                pipeline,
                records,
                cfg,
                name,
                k=k,
                with_hallucination=with_hallucination and shared_rate is None,
                max_workers=workers,
                hallucination_rate=shared_rate,
            )
        )
    return results
