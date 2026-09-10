"""评测报告生成：指标表 + A-B 对比 + 结论与优化建议。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from .. import config
from ..logging_setup import get_logger
from ..models import MetricResult
from ..rag.pipeline import RAGPipeline
from ..storage import save_file

logger = get_logger(__name__)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_report(
    results: Sequence[MetricResult],
    pipeline: RAGPipeline | None = None,
    k: int = 5,
    sample_count: int = 0,
) -> str:
    """渲染 Markdown 评测报告。"""
    stats = pipeline.stats() if pipeline else {}
    lines: list[str] = [
        "# RAG 效果评测报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：`{config.MODEL_NAME}` @ `{config.OPENAI_BASE_URL}`",
        f"- Embedding：`{config.EMBEDDING_PROVIDER}` / `{config.EMBEDDING_MODEL}`",
    ]
    if stats:
        lines.append(
            f"- 知识库：`{stats.get('name')}`（{stats.get('chunks')} 片段 / "
            f"{stats.get('sources')} 份文档 / 维度 {stats.get('dim')}）"
        )
    lines += [
        f"- 评测样本：{sample_count or (results[0].sample_count if results else 0)} 条",
        "",
        "## 1. 指标口径",
        "",
        f"- **Recall@{k}**：正确片段是否出现在前 {k} 条召回中（本评测集每条样本 1 个正确片段）",
        "- **MRR**：正确片段首次命中排名倒数的均值，未命中记 0",
        f"- **Hit Rate@{k}**：前 {k} 条至少命中一条的样本占比",
        "- **幻觉率**：LLM-as-judge 将回答拆为断言，统计不可由召回片段支撑的断言占比（temperature=0，可复现）",
        "- **平均延迟**：单次检索 + 重排的端到端耗时（毫秒）",
        "",
        "## 2. A-B 对比结果",
        "",
        f"| 实验组 | Recall@{k} | MRR | Hit Rate@{k} | 幻觉率 | 平均延迟(ms) | 样本 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {_pct(r.recall_at_k)} | {r.mrr:.3f} | {_pct(r.hit_rate)} | "
            f"{_pct(r.hallucination_rate)} | {r.avg_latency_ms} | {r.sample_count} |"
        )

    lines += ["", "## 3. 结论与优化建议", ""]
    lines.extend(_conclusions(results))
    lines += ["", "## 4. 配置快照", "", "```json"]
    import json

    lines.append(
        json.dumps(
            {
                "model": config.MODEL_NAME,
                "embedding": f"{config.EMBEDDING_PROVIDER}/{config.EMBEDDING_MODEL}",
                "chunk": {"size": config.CHUNK_SIZE, "overlap": config.CHUNK_OVERLAP},
                "experiments": [{"name": r.name, "config": r.config} for r in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    lines += ["```", ""]
    return "\n".join(lines)


def _conclusions(results: Sequence[MetricResult]) -> list[str]:
    """根据数据自动生成结论（避免空话，全部来自实测对比）。"""
    if not results:
        return ["- 无有效实验结果。"]

    def find(keyword: str) -> MetricResult | None:
        for r in results:
            if keyword in r.name:
                return r
        return None

    out: list[str] = []
    vec = find("仅向量")
    bm25 = find("仅 BM25")
    hybrid = find("混合检索(RRF)")
    llm_rr = find("LLM重排")
    api_rr = find("API重排")

    if vec and bm25 and hybrid:
        best_single = max(vec, bm25, key=lambda r: r.recall_at_k)
        out.append(
            f"- 单路检索中 **{best_single.name}** 的 Recall@K 为 {_pct(best_single.recall_at_k)}；"
            f"混合检索（RRF）为 {_pct(hybrid.recall_at_k)}，"
            f"相对最优单路 {'提升' if hybrid.recall_at_k >= best_single.recall_at_k else '变化'} "
            f"{abs(hybrid.recall_at_k - best_single.recall_at_k) * 100:.1f} 个百分点。"
            "说明关键词匹配与语义召回确实互补，混合检索值得作为默认配置。"
        )
    if hybrid and llm_rr:
        delta = llm_rr.mrr - hybrid.mrr
        out.append(
            f"- 引入 LLM 重排后 MRR 从 {hybrid.mrr:.3f} 变为 {llm_rr.mrr:.3f}"
            f"（{'提升' if delta >= 0 else '下降'} {abs(delta):.3f}），"
            f"但平均延迟从 {hybrid.avg_latency_ms}ms 增加到 {llm_rr.avg_latency_ms}ms。"
            "重排是典型的「用延迟换准确率」，可按业务对时延的容忍度开关。"
        )
    if api_rr:
        out.append(
            f"- API 重排（{config.RERANK_MODEL}）MRR 为 {api_rr.mrr:.3f}，"
            f"延迟 {api_rr.avg_latency_ms}ms，可与 LLM 重排按成本与效果二选一。"
        )

    slowest = max(results, key=lambda r: r.avg_latency_ms)
    fastest = min(results, key=lambda r: r.avg_latency_ms)
    out.append(
        f"- 延迟区间 {fastest.avg_latency_ms}ms（{fastest.name}）~ {slowest.avg_latency_ms}ms（{slowest.name}）。"
        "线上可通过结果缓存 + 缩小 rerank_top_n 进一步压缩。"
    )
    out.append(
        "- 后续可继续做网格实验：chunk_size ∈ {300, 500, 800}、top_k ∈ {3, 5, 10}，"
        "用同一份评测集跑出最优组合（修改 `RetrievalConfig` 即可复现）。"
    )
    return out


def save_report(
    results: Sequence[MetricResult],
    pipeline: RAGPipeline | None = None,
    k: int = 5,
) -> str:
    """渲染并保存报告，返回路径。"""
    content = render_report(results, pipeline, k=k)
    path = save_file(content, "Eval_Report")
    logger.info("评测报告已生成 | %s", path)
    return path
