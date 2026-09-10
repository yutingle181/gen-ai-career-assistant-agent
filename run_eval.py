"""一键评测脚本：建库 -> 生成评测集 -> 跑 A-B 实验 -> 产出 Markdown 报告。

用法：
    .venv\\Scripts\\python.exe run_eval.py                  # 用示例知识库
    .venv\\Scripts\\python.exe run_eval.py --kb mykb        # 指定知识库
    .venv\\Scripts\\python.exe run_eval.py --no-judge       # 跳过幻觉判定（更快更省）
    .venv\\Scripts\\python.exe run_eval.py --workers 8      # 提高并发（默认 EVAL_MAX_WORKERS）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.eval import (  # noqa: E402
    build_offline_records,
    default_experiments,
    generate_from_pipeline,
    load_records,
    run_all,
    save_records,
    save_report,
)
from src.logging_setup import get_logger  # noqa: E402
from src.rag import get_registry  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
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
    args = parser.parse_args()

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

    experiments = default_experiments(include_api_rerank=args.api_rerank)
    workers = args.workers if args.workers > 0 else config.EVAL_MAX_WORKERS
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

    report_path = save_report(results, pipeline, k=args.top_k)
    print(f"\n评测报告已生成：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
