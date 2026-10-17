"""把 locust 的阶梯压测 CSV 画成曲线（P50 / P95 / P99 + 吞吐）。

为什么要有这一步：一张表里的三行数字看不出"拐点在哪"，
而拐点才是压测唯一有价值的结论 ✓ —— 所以图上会直接标出
「延迟开始明显上升」的那一档 ✓。

用法：
    .venv-ci\\Scripts\\python.exe bench\\plot_bench.py --results-dir bench/results
    .venv-ci\\Scripts\\python.exe bench\\plot_bench.py --results-dir bench/results --out bench/results/latency.png

只依赖 matplotlib（开发工具，不进 requirements ✓）——
压测是手工活，不该让 CI 每次都去装绘图库 ✓。
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示器环境（CI/服务器）也要能出图
import matplotlib.pyplot as plt  # noqa: E402

# 图内文字一律用英文：matplotlib 默认字体（DejaVu Sans）没有中文字形，
# 写中文会渲染成一片"豆腐块"（实测一堆 Glyph missing from font 警告）。
# 表格与文档仍是中文——只有图里的字需要避开字体缺失。
FONT = {"family": "DejaVu Sans"}


def _read_stage(csv_path: Path) -> dict:
    """读一个阶段的 CSV，返回 {场景: {P50, P95, P99, RPS, 失败率}}。

    locust 的 `*_stats.csv` 每行是一个请求名的聚合统计 ✓，
    比自己从历史 CSV 重算更可信（用的是 locust 自己的分位数实现 ✓）。
    """
    rows: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Name") or row.get("Type") or "").strip()
            if not name or name == "Aggregated":
                continue

            def _f(key: str, default: float = 0.0, _row=row) -> float:
                # 显式绑定 row：闭包捕获循环变量是 ruff B023，
                # 这里是立即调用所以行为上没问题，但绑定后意图明确、也不会被后续重构坑到。
                try:
                    return float(_row.get(key) or default)
                except (TypeError, ValueError):
                    return default

            reqs = _f("Request Count")
            fails = _f("Failure Count")
            rows[name] = {
                "P50": _f("50%"),
                "P95": _f("95%"),
                "P99": _f("99%"),
                "avg": _f("Average"),
                "rps": _f("Requests/s"),
                "fail_rate": (fails / reqs * 100) if reqs else 0.0,
                "reqs": reqs,
            }
    return rows


def collect(results_dir: Path) -> tuple[dict, list[int]]:
    """把所有 <场景>-<并发>_stats.csv 汇总成 {场景: {并发数: 指标}}。"""
    by_scenario: dict[str, dict[int, dict]] = defaultdict(dict)
    stages: set[int] = set()
    # locust 会把 --csv 前缀后面各加一段后缀：
    #   <前缀>_stats.csv / <前缀>_stats_history.csv / <前缀>_exceptions.csv / <前缀>_failures.csv
    # 所以本项目 `baseline-10.csv` 前缀最终产出的是 `baseline-10.csv_stats.csv` ✓。
    # 只读 stats 那一类：history 也有 Name 字段，混进来会把正确统计**覆盖成空**（实测踩过）。
    for path in sorted(results_dir.glob("*-*.csv_stats.csv")):
        m = re.match(r"(?P<scenario>[a-z_]+)-(?P<stage>\d+)\.csv_stats\.csv$", path.name)
        if not m:
            continue
        scenario_key = m.group("scenario")
        stage = int(m.group("stage"))
        stages.add(stage)
        for _name, metrics in _read_stage(path).items():
            by_scenario[scenario_key][stage] = metrics
    return by_scenario, sorted(stages)


def plot(by_scenario: dict, stages: list[int], out: Path, title: str) -> None:
    if not by_scenario or not stages:
        print("没有可用的 CSV —— 先跑 run_bench.ps1")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    colors = {"P50": "#4C78A8", "P95": "#F58518", "P99": "#E45756"}
    for scenario, per_stage in sorted(by_scenario.items()):
        xs = [s for s in stages if s in per_stage]
        if not xs:
            continue
        for pct in ("P50", "P95", "P99"):
            axes[0].plot(
                xs,
                [per_stage[x][pct] for x in xs],
                marker="o",
                color=colors[pct],
                linestyle="-" if pct == "P95" else "--",
                label=f"{scenario} · {pct}",
            )
        axes[1].plot(
            xs,
            [per_stage[x]["rps"] for x in xs],
            marker="s",
            label=scenario,
        )

    axes[0].set_xlabel("concurrency (users)", fontdict=FONT)
    axes[0].set_ylabel("latency (ms)", fontdict=FONT)
    axes[0].set_title("Latency percentiles  (knee = where P95 starts climbing)", fontdict=FONT)
    axes[0].set_xticks(stages)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xlabel("concurrency (users)", fontdict=FONT)
    axes[1].set_ylabel("throughput (req/s)", fontdict=FONT)
    axes[1].set_title("Throughput  (falls as concurrency rises = past the knee)", fontdict=FONT)
    axes[1].set_xticks(stages)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, prop={"family": "DejaVu Sans", "size": 7})

    fig.suptitle(title, fontsize=11, fontdict=FONT)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"图已生成：{out}")


def print_table(by_scenario: dict, stages: list[int]) -> None:
    print("\n===== 阶梯压测汇总 =====")
    header = f"{'场景':<34}{'并发':>5}{'P50':>9}{'P95':>9}{'P99':>9}{'吞吐':>9}{'失败率':>9}"
    print(header)
    print("-" * len(header.encode("gbk", errors="ignore")))
    for scenario, per_stage in sorted(by_scenario.items()):
        for stage in stages:
            m = per_stage.get(stage)
            if not m:
                continue
            print(
                f"{scenario:<34}{stage:>5}{m['P50']:>9.0f}{m['P95']:>9.0f}"
                f"{m['P99']:>9.0f}{m['rps']:>9.1f}{m['fail_rate']:>8.1f}%"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="画 locust 阶梯压测曲线")
    parser.add_argument("--results-dir", default="bench/results")
    parser.add_argument("--out", default="")
    parser.add_argument("--title", default="GenAI Career Assistant - staged load test (single process, local)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    by_scenario, stages = collect(results_dir)
    print(f"读到阶段：{stages}")
    print(f"读到场景：{sorted(by_scenario)}")
    print_table(by_scenario, stages)

    if args.out:
        out = Path(args.out)
    else:
        out = results_dir / "latency.png"
    plot(by_scenario, stages, out, args.title)

    print("\n口径提醒：")
    print("  · baseline（GET /）的 QPS 只说明框架不是瓶颈，不代表真实业务吞吐；")
    print("  · 本机单进程结果不能当线上容量结论，线上还要叠加网关/网络/多副本；")
    print("  · 未开启 -IncludeChat 时不含 LLM 路径（那部分会烧 token，且时延由模型主导）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
