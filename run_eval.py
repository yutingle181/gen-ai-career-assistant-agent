"""一键评测脚本：建库 -> 生成评测集 -> 跑 A-B 实验 -> 产出 Markdown 报告。

用法：
    .venv\\Scripts\\python.exe run_eval.py                  # 用示例知识库
    .venv\\Scripts\\python.exe run_eval.py --kb mykb        # 指定知识库
    .venv\\Scripts\\python.exe run_eval.py --no-judge       # 跳过幻觉判定（更快更省）
    .venv\\Scripts\\python.exe run_eval.py --workers 8      # 提高并发（默认 EVAL_MAX_WORKERS）
    .venv\\Scripts\\python.exe run_eval.py --tool-ab        # 额外跑「显式检索 vs Function Calling」A/B
    .venv\\Scripts\\python.exe run_eval.py --tool-ab --tool-ab-model qwen-turbo --tool-ab-samples 15

说明：`--tool-ab` 需要可用的 API Key；两条路径会钉在同一模型上（默认 `MODEL_NAME`），
否则对比到的是「模型差异」而不是「链路差异」。A/B 失败不会影响主报告的生成。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.eval import (  # noqa: E402
    ToolComparison,
    build_offline_records,
    default_experiments,
    generate_from_pipeline,
    load_records,
    run_all,
    run_tool_ab,
    save_records,
    save_report,
)
from src.logging_setup import get_logger  # noqa: E402
from src.rag import get_registry  # noqa: E402

logger = get_logger(__name__)


def _print_tool_ab(comparison: ToolComparison) -> None:
    """在终端打印双路径对比表与关键结论（与报告章节同口径）。"""
    print("\n" + "=" * 72)
    print(f"工具调用 A/B 对比（显式检索 vs Function Calling，{comparison.sample_count} 条样本）")
    print("=" * 72)
    print(
        f"{'路径':<18}{'成功率':>8}{'平均延迟ms':>12}{'P95ms':>10}"
        f"{'平均token':>11}{'平均轮次':>10}{'调用率':>8}"
    )
    print("-" * 72)
    for label, m in comparison.rows():
        print(
            f"{label:<18}{m.success_rate * 100:>7.1f}%{m.avg_latency_ms:>12.0f}"
            f"{m.p95_latency_ms:>10.0f}{m.avg_tokens:>11.0f}"
            f"{m.avg_rounds:>10.2f}{m.tool_call_rate * 100:>7.1f}%"
        )
    print("-" * 72)

    base, tool = comparison.explicit, comparison.tool_calling
    if base.avg_latency_ms:
        delta = (tool.avg_latency_ms - base.avg_latency_ms) / base.avg_latency_ms * 100
        print(f"延迟变化：{delta:+.1f}%（负值表示 Function Calling 更快）")
    if base.avg_tokens:
        print(f"token 倍数：{tool.avg_tokens / base.avg_tokens:.2f}x（相对显式检索）")
    print(
        f"自主检索比例：{tool.tool_call_rate * 100:.1f}% 的样本由模型决定调用工具"
        "（显式路径恒为 100%）。口径：tiktoken 近似计数，衡量链路相对成本，不含答案质量。"
    )
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    单独抽出来是为了让「参数契约」可被离线单测直接断言：测试不必启动建库、
    跑批与报告落盘等重流程，也能锁住参数名、默认值与 help 文案。
    """
    parser = argparse.ArgumentParser(description="RAG 效果评测")
    parser.add_argument("--kb", default="samples", help="知识库名称")
    parser.add_argument("--source", default=str(config.SAMPLES_DIR), help="建库来源目录")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    parser.add_argument("--no-judge", action="store_true", help="跳过幻觉率判定")
    parser.add_argument("--api-rerank", action="store_true", help="额外跑 API 重排实验组")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=f"评测并发线程数，默认取 config.EVAL_MAX_WORKERS（当前 {config.EVAL_MAX_WORKERS}），1 表示串行",
    )
    parser.add_argument(
        "--tool-ab",
        action="store_true",
        help="额外跑「显式检索 vs Function Calling」双路径 A/B（需可用 API Key，耗时更长）",
    )
    parser.add_argument(
        "--tool-ab-model",
        default="",
        help="钉住 A/B 两条路径使用的模型，默认取 config.MODEL_NAME（如 qwen-turbo）",
    )
    parser.add_argument(
        "--tool-ab-samples",
        type=int,
        default=0,
        help="A/B 只取前 N 条样本，默认全部（用于控制成本）",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="A/B 时不装配联网检索工具（仅知识库检索，更省更稳）",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="跳过 4 组检索实验，只跑工具 A/B（故障注入等场景下省时省 token）",
    )
    parser.add_argument(
        "--fault-tool",
        default="",
        help="故障注入：给哪个工具注入模拟故障（knowledge_search / search_web），留空即关闭",
    )
    parser.add_argument(
        "--fault-kind",
        default="http_5xx",
        help="注入的故障类型：http_5xx / http_4xx / timeout / empty（empty=正常返回但没结果）",
    )
    parser.add_argument(
        "--fault-rate",
        type=float,
        default=0.0,
        help="每次工具调用的失败概率（0~1）；不填或 0 视为 1.0（全失败）",
    )
    parser.add_argument(
        "--fault-seed",
        type=int,
        default=0,
        help="注入随机种子（固定后同批输入复现同一串故障；注入实验请配 --workers 1）",
    )
    return parser


def apply_fault_injection(args) -> None:
    """把 CLI 的注入参数落到 config，并打一条显眼警告。

    之所以要在建库/评测之前就设好：报告快照会记录注入配置。
    「这份报告是在 100% 知识库检索失败下跑出来的」必须写在报告里，
    否则读者会把降级数据当成系统真实水平 —— 那是比没有数据更糟的事。
    """
    if not getattr(args, "fault_tool", ""):
        return
    config.ENABLE_FAULT_INJECTION = True
    config.FAULT_INJECT_TOOL = args.fault_tool
    config.FAULT_INJECT_KIND = args.fault_kind
    config.FAULT_INJECT_RATE = args.fault_rate if args.fault_rate > 0 else 1.0
    if args.fault_seed:
        config.FAULT_INJECT_SEED = args.fault_seed
    print(
        "\n" + "*" * 72
        + f"\n故障注入已启用：工具={config.FAULT_INJECT_TOOL} 类型={config.FAULT_INJECT_KIND} "
        f"概率={config.FAULT_INJECT_RATE:.0%} 种子={config.FAULT_INJECT_SEED}"
        "\n这是用于验证**失败路径**的实验，数据反映降级行为，不是系统真实水平"
        "\n" + "*" * 72 + "\n"
    )


def main() -> int:
    args = build_parser().parse_args()
    apply_fault_injection(args)

    registry = get_registry()
    pipeline = registry.get_or_create(args.kb)

    if args.rebuild or not pipeline.ready:
        logger.info("开始建库：%s <- %s", args.kb, args.source)
        result = pipeline.ingest([args.source])
        logger.info(
            "建库完成：文档 %d / 片段 %d / 维度 %d / 耗时 %dms",
            result.documents, result.chunks, result.dim, result.elapsed_ms,
        )
        if result.chunks == 0:
            print("建库未产生片段，请检查 --source 目录下是否有支持的文档。")
            return 1

    # 评测集
    records = load_records()
    if not records:
        logger.info("未发现评测集，尝试用 LLM 自动生成…")
        try:
            records = generate_from_pipeline(pipeline, per_chunk=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 生成失败：%s", exc)
        if not records:
            logger.warning("回退为离线方式构造评测集（质量较低，仅供流程验证）")
            records = build_offline_records(pipeline)
        if records:
            path = save_records(records)
            print(f"评测集已生成：{path}（{len(records)} 条）")

    if not records:
        print("评测集为空，无法评测。")
        return 1

    workers = args.workers if args.workers > 0 else config.EVAL_MAX_WORKERS
    results: list = []
    if args.skip_rag:
        # 故障注入实验只关心工具链路的降级行为，检索实验组（4 组 × 全量样本 + 幻觉判定）
        # 既慢又费 token，因此允许单独跑 A/B；报告会如实标注本次不含检索质量表。
        print("\n--skip-rag：跳过检索实验组，只跑工具 A/B（本次报告不含检索质量表）")
    else:
        experiments = default_experiments(include_api_rerank=args.api_rerank)
        print(f"\n开始跑 {len(experiments)} 组实验，共 {len(records)} 条样本，并发 {workers}…\n")
        results = run_all(
            pipeline,
            records,
            experiments,
            k=args.top_k,
            with_hallucination=not args.no_judge,
            max_workers=workers,
        )

        print("\n" + "=" * 72)
        print(f"{'实验组':<24}{'Recall@' + str(args.top_k):>10}{'MRR':>10}{'HitRate':>10}{'延迟ms':>10}")
        print("=" * 72)
        for r in results:
            print(
                f"{r.name:<24}{r.recall_at_k * 100:>9.1f}%{r.mrr:>10.3f}"
                f"{r.hit_rate * 100:>9.1f}%{r.avg_latency_ms:>10}"
            )
        print("=" * 72)

    tool_comparison = None
    if args.tool_ab:
        ab_records = records[: args.tool_ab_samples] if args.tool_ab_samples > 0 else records
        used_model = args.tool_ab_model or config.MODEL_NAME
        print(
            f"\n开始跑双路径 A/B：{len(ab_records)} 条样本 / 模型 {used_model}"
            f" / 联网检索 {'关闭' if args.no_web else '开启'}…"
        )
        try:
            tool_comparison = run_tool_ab(
                pipeline,
                ab_records,
                kb_name=args.kb,
                max_workers=workers,
                model=args.tool_ab_model or None,
                include_web=not args.no_web,
            )
            _print_tool_ab(tool_comparison)
        except Exception as exc:  # noqa: BLE001
            # A/B 失败不拖垮主报告：降级为「仅检索质量报告」并如实提示
            logger.warning("工具调用 A/B 未完成：%s", exc)
            print(f"工具调用 A/B 未完成（{type(exc).__name__}）：{exc}")

    report_path = save_report(results, pipeline, k=args.top_k, tool_comparison=tool_comparison)
    print(f"\n评测报告已生成：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
