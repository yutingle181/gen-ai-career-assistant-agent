"""离线评测体系：数据集 / 指标 / 跑批 / 报告。"""

from .dataset import build_offline_records, generate_from_pipeline, load_records, save_records
from .metrics import evaluate_retrieval, judge_hallucination, recall_at_k
from .report import render_report, render_tool_comparison, save_report
from .runner import default_experiments, run_all, run_experiment
from .tool_eval import (
    PathOutcome,
    ToolComparison,
    ToolPathMetric,
    compare_paths,
    count_tokens,
    make_explicit_runner,
    make_tool_runner,
    run_tool_ab,
    summarize_path,
)

__all__ = [
    "generate_from_pipeline",
    "build_offline_records",
    "load_records",
    "save_records",
    "recall_at_k",
    "evaluate_retrieval",
    "judge_hallucination",
    "default_experiments",
    "run_experiment",
    "run_all",
    "render_report",
    "render_tool_comparison",
    "save_report",
    "PathOutcome",
    "ToolPathMetric",
    "ToolComparison",
    "summarize_path",
    "compare_paths",
    "make_explicit_runner",
    "make_tool_runner",
    "run_tool_ab",
    "count_tokens",
]
