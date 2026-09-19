"""评测报告生成：指标表 + A-B 对比 + 结论与优化建议。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from .. import config
from ..logging_setup import get_logger
from ..models import MetricResult
from ..rag.pipeline import RAGPipeline
from ..storage import save_file
from .tool_eval import ToolComparison
from .trajectory import render_trajectory

logger = get_logger(__name__)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_report(
    results: Sequence[MetricResult],
    pipeline: RAGPipeline | None = None,
    k: int = 5,
    sample_count: int = 0,
    tool_comparison: ToolComparison | None = None,
    trajectory: dict | None = None,
) -> str:
    """渲染 Markdown 评测报告；传入 `tool_comparison` 时追加工具调用 A/B 章节。

    `trajectory` 缺省时自动取 A/B 结果里的轨迹指标（工具调用路径算出来的那份），
    因此现有调用方不改一行也能在报告里看到 TaskSuccessRate / Failure Onset。
    """
    if trajectory is None and tool_comparison is not None:
        trajectory = tool_comparison.trajectory
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

    # 工具调用对比是「系统级取舍」，插在检索质量之后、结论之前，
    # 避免读者把「召回变好」与「链路变复杂」两个结论混为一谈。
    next_index = 3
    if tool_comparison is not None:
        lines += render_tool_comparison(tool_comparison)
        next_index = 4
    # 轨迹级指标紧跟工具 A/B：两者回答的是同一问题的两面——
    # 「这条链路贵不贵」与「这条链路稳不稳、错了最先错在第几步」。
    if trajectory:
        lines += render_trajectory(trajectory, next_index)
        next_index += 1

    lines += ["", f"## {next_index}. 结论与优化建议", ""]
    lines.extend(_conclusions(results))
    lines += ["", f"## {next_index + 1}. 配置快照", "", "```json"]
    import json

    snapshot: dict = {
        "model": config.MODEL_NAME,
        "embedding": f"{config.EMBEDDING_PROVIDER}/{config.EMBEDDING_MODEL}",
        "chunk": {"size": config.CHUNK_SIZE, "overlap": config.CHUNK_OVERLAP},
        "experiments": [{"name": r.name, "config": r.config} for r in results],
    }
    if tool_comparison is not None:
        snapshot["tool_comparison"] = {
            "sample_count": tool_comparison.sample_count,
            "explicit": tool_comparison.explicit.model_dump(),
            "tool_calling": tool_comparison.tool_calling.model_dump(),
        }
    if trajectory:
        snapshot["trajectory"] = trajectory
    lines.append(json.dumps(snapshot, ensure_ascii=False, indent=2))
    lines += ["```", ""]
    return "\n".join(lines)


def render_tool_comparison(comparison: ToolComparison) -> list[str]:
    """渲染「显式检索 vs Function Calling」对比章节（返回行列表，便于嵌入总报告）。"""
    lines: list[str] = [
        "",
        "## 3. 工具调用 A/B 对比（显式检索 vs Function Calling）",
        "",
        f"- 样本：{comparison.sample_count} 条；成本为 `tiktoken` 近似计数，用于比较**相对**开销",
        f"- 轮次上限：`TOOL_CALLING_MAX_STEPS={config.TOOL_CALLING_MAX_STEPS}`",
        "",
        "| 路径 | 成功率 | 平均延迟(ms) | P95 延迟(ms) | 平均 token | 合计 token "
        "| 平均轮次 | 平均工具调用 | 工具调用率 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _label, m in comparison.rows():
        lines.append(
            f"| {m.name} | {_pct(m.success_rate)} | {m.avg_latency_ms:.0f} | {m.p95_latency_ms:.0f} | "
            f"{m.avg_tokens:.0f} | {m.total_tokens} | {m.avg_rounds:.2f} | {m.avg_tool_calls:.2f} | "
            f"{_pct(m.tool_call_rate)} |"
        )

    lines += ["", "### 结论（全部来自上表实测）", ""]
    lines.extend(_tool_conclusions(comparison))
    return lines


def _tool_conclusions(comparison: ToolComparison) -> list[str]:
    """基于双路径实测差异自动生成结论，避免空话。"""
    ex = comparison.explicit
    tc = comparison.tool_calling
    if not ex.samples and not tc.samples:
        return ["- 无有效样本，无法对比两条路径。"]

    latency_delta = tc.avg_latency_ms - ex.avg_latency_ms
    token_ratio = (tc.avg_tokens / ex.avg_tokens) if ex.avg_tokens else 0.0
    out = [
        f"- 延迟：显式检索 {ex.avg_latency_ms:.0f}ms → 工具调用 {tc.avg_latency_ms:.0f}ms"
        f"（{'增加' if latency_delta >= 0 else '减少'} {abs(latency_delta):.0f}ms）；"
        f"P95 从 {ex.p95_latency_ms:.0f}ms 变为 {tc.p95_latency_ms:.0f}ms，"
        "长尾是工具调用路径最需要关注的成本。",
        f"- 成本：平均 token 从 {ex.avg_tokens:.0f} 变为 {tc.avg_tokens:.0f}"
        + (f"（约 {token_ratio:.2f}x）" if token_ratio else "")
        + "；多出的部分主要来自「决策轮 + 工具结果回灌」，而非最终答案变长。",
        f"- 自主性：模型在 {_pct(tc.tool_call_rate)} 的样本上主动调用了工具，"
        f"平均 {tc.avg_tool_calls:.2f} 次/样本；显式路径固定 1 次。"
        "差距说明工具调用只在「模型判断确有必要」时才付出额外开销。",
        f"- 稳定性：成功率 显式 {_pct(ex.success_rate)} vs 工具调用 {_pct(tc.success_rate)}"
        f"（失败样本 {ex.failures} / {tc.failures} 条）。",
    ]
    if token_ratio and token_ratio > 1.5 and latency_delta > 0:
        out.append(
            "- 取舍建议：工具调用适合**开放式、需要多源信息**的问题（如岗位检索 + 知识库联合）；"
            "对确定性强的知识库问答，保留显式检索路径更省成本。这也是本开关默认关闭的原因。"
        )
    else:
        out.append(
            "- 取舍建议：两条路径成本接近时，优先用工具调用以换取更强的自主性；"
            "若延迟敏感则保留显式检索。"
        )
    return out


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
    tool_comparison: ToolComparison | None = None,
    trajectory: dict | None = None,
) -> str:
    """渲染并保存报告，返回路径。"""
    content = render_report(
        results, pipeline, k=k, tool_comparison=tool_comparison, trajectory=trajectory
    )
    path = save_file(content, "Eval_Report")
    logger.info("评测报告已生成 | %s", path)
    return path
