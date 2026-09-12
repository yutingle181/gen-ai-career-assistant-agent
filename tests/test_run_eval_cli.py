"""`run_eval.py` 命令行入口的离线单测：锁住参数契约与 A/B 分流 / 降级行为。

设计要点：
- 只测 `run_eval` 的**参数与分支语义**，不测打印文案的逐字写法（文案会随措辞调整）。
- 通过 `monkeypatch` 打桩 `run_eval.<name>`（这些名字在 `run_eval` 顶层被 import 绑定，
  打桩必须落在 `run_eval` 模块属性上，patch `src.eval.*` 不会生效）。
- 全程无网络、无模型调用、无真实落盘，可在 CI 秒级跑完。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import run_eval
from src import config
from src.eval import ToolComparison, ToolPathMetric
from src.models import EvalRecord


def _records(count: int = 5) -> list[EvalRecord]:
    return [EvalRecord(question=f"q{i}", golden_chunk_ids=[f"c{i}"]) for i in range(count)]


def _results() -> list[SimpleNamespace]:
    """伪造 4 组实验汇总结果（打印逻辑会读取这 5 个属性）。"""
    return [
        SimpleNamespace(
            name=f"实验组{i}",
            recall_at_k=0.5 + i * 0.1,
            mrr=0.4 + i * 0.05,
            hit_rate=0.6,
            avg_latency_ms=100 + i * 10,
        )
        for i in range(4)
    ]


def _comparison() -> ToolComparison:
    """真实的 `ToolComparison`，让 `_print_tool_ab` 的计算与渲染走真代码。"""
    return ToolComparison(
        explicit=ToolPathMetric(
            name="显式检索",
            samples=2,
            success_rate=1.0,
            avg_latency_ms=2000.0,
            p95_latency_ms=2500.0,
            avg_tokens=1000.0,
            avg_rounds=1.0,
            tool_call_rate=1.0,
        ),
        tool_calling=ToolPathMetric(
            name="Function Calling",
            samples=2,
            success_rate=1.0,
            avg_latency_ms=1000.0,
            p95_latency_ms=1200.0,
            avg_tokens=500.0,
            avg_rounds=1.5,
            tool_call_rate=0.5,
        ),
        sample_count=2,
    )


def _install_stubs(
    monkeypatch,
    *,
    records: list[EvalRecord] | None = None,
    tool_ab_error: Exception | None = None,
) -> SimpleNamespace:
    """把 `main()` 的重依赖全部替换为可观测的假实现，并记录调用参数。"""
    calls = SimpleNamespace(run_all=[], save_report=[], run_tool_ab=[])
    pipeline = SimpleNamespace(ready=True)
    registry = SimpleNamespace(get_or_create=lambda name: pipeline)

    def fake_get_registry():
        return registry

    def fake_load_records(*args, **kwargs):
        return list(records if records is not None else _records())

    def fake_run_all(*args, **kwargs):
        calls.run_all.append(kwargs)
        return _results()

    def fake_save_report(results, pipeline_arg, **kwargs):
        calls.save_report.append(kwargs)
        return "Agent_output/Eval_Report_test.md"

    def fake_run_tool_ab(*args, **kwargs):
        calls.run_tool_ab.append({"args": args, "kwargs": kwargs})
        if tool_ab_error is not None:
            raise tool_ab_error
        return _comparison()

    monkeypatch.setattr(run_eval, "get_registry", fake_get_registry)
    monkeypatch.setattr(run_eval, "load_records", fake_load_records)
    monkeypatch.setattr(run_eval, "run_all", fake_run_all)
    monkeypatch.setattr(run_eval, "save_report", fake_save_report)
    monkeypatch.setattr(run_eval, "run_tool_ab", fake_run_tool_ab)
    return calls


# ---------------------------------------------------------------- 参数契约
def test_build_parser_defaults():
    """不传任何参数时，各默认值就是对外承诺的契约。"""
    args = run_eval.build_parser().parse_args([])
    assert args.kb == "samples"
    assert args.source == str(config.SAMPLES_DIR)
    assert args.rebuild is False
    assert args.no_judge is False
    assert args.api_rerank is False
    assert args.top_k == 5
    assert args.workers == 0  # 0 表示回退到 config.EVAL_MAX_WORKERS
    assert args.tool_ab is False
    assert args.tool_ab_model == ""
    assert args.tool_ab_samples == 0
    assert args.no_web is False
    print("[OK] 默认参数契约一致")


def test_build_parser_explicit_values():
    """显式传参（尤其 A/B 四个参数）必须被正确解析。"""
    args = run_eval.build_parser().parse_args(
        [
            "--kb", "mykb",
            "--source", "data/custom",
            "--rebuild",
            "--no-judge",
            "--api-rerank",
            "--top-k", "8",
            "--workers", "4",
            "--tool-ab",
            "--tool-ab-model", "qwen-turbo",
            "--tool-ab-samples", "15",
            "--no-web",
        ]
    )
    assert args.kb == "mykb"
    assert args.source == "data/custom"
    assert args.rebuild is True
    assert args.no_judge is True
    assert args.api_rerank is True
    assert args.top_k == 8
    assert args.workers == 4
    assert args.tool_ab is True
    assert args.tool_ab_model == "qwen-turbo"
    assert args.tool_ab_samples == 15
    assert args.no_web is True
    print("[OK] 显式参数解析正确")


def test_build_parser_help_renders_worker_default(capsys):
    """`--workers` 的 help 里嵌了 EVAL_MAX_WORKERS 当前值，抽函数后必须仍然渲染。"""
    with pytest.raises(SystemExit) as exc:
        run_eval.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "EVAL_MAX_WORKERS" in out
    assert str(config.EVAL_MAX_WORKERS) in out
    assert "--tool-ab-samples" in out
    print("[OK] --help 正常渲染（含 workers 默认值）")


# ---------------------------------------------------------------- A/B 分流
def test_main_without_tool_ab_keeps_report_only(monkeypatch):
    """未开 `--tool-ab` 时不得触碰 A/B，报告仍按 tool_comparison=None 生成。"""
    calls = _install_stubs(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])

    assert run_eval.main() == 0
    assert calls.run_tool_ab == [], "未开 --tool-ab 时不应调用 run_tool_ab"
    assert len(calls.save_report) == 1
    assert calls.save_report[0]["tool_comparison"] is None
    assert calls.save_report[0]["k"] == 5
    assert len(calls.run_all) == 1
    print("[OK] 未开 --tool-ab：跳过 A/B 且报告降级为纯检索")


def test_main_tool_ab_truncates_samples_and_passes_flags(monkeypatch):
    """开启 A/B：样本按 --tool-ab-samples 截断，--no-web / --workers 正确传导。"""
    calls = _install_stubs(monkeypatch, records=_records(5))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_eval.py", "--tool-ab", "--tool-ab-samples", "3", "--no-web", "--workers", "6", "--top-k", "7"],
    )

    assert run_eval.main() == 0
    assert len(calls.run_tool_ab) == 1
    call = calls.run_tool_ab[0]
    ab_records = call["args"][1]
    assert len(ab_records) == 3, "应只取前 3 条样本"
    assert call["kwargs"]["kb_name"] == "samples"
    assert call["kwargs"]["max_workers"] == 6
    assert call["kwargs"]["model"] is None, "未指定 --tool-ab-model 时应传 None 让下游取 config.MODEL_NAME"
    assert call["kwargs"]["include_web"] is False
    assert calls.save_report[0]["tool_comparison"] is not None, "A/B 成功时对比结果应写入报告"
    print("[OK] A/B 样本截断与参数传导正确")


def test_main_tool_ab_pins_model_and_keeps_web(monkeypatch):
    """指定模型时钉住模型；不带 --no-web 时联网检索保持开启、样本不截断。"""
    calls = _install_stubs(monkeypatch, records=_records(4))
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--tool-ab", "--tool-ab-model", "qwen-turbo"])

    assert run_eval.main() == 0
    call = calls.run_tool_ab[0]
    assert len(call["args"][1]) == 4, "未指定 --tool-ab-samples 时应使用全部样本"
    assert call["kwargs"]["model"] == "qwen-turbo"
    assert call["kwargs"]["include_web"] is True
    print("[OK] A/B 模型钉住与联网开关正确")


def test_main_tool_ab_failure_degrades_without_breaking_report(monkeypatch, capsys):
    """A/B 抛异常必须被吞掉：主报告照常生成、返回码仍为 0、对比结果不写入报告。"""
    calls = _install_stubs(monkeypatch, tool_ab_error=RuntimeError("模拟 A/B 崩溃"))
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--tool-ab"])

    assert run_eval.main() == 0
    out = capsys.readouterr().out
    assert "工具调用 A/B 未完成" in out
    assert "RuntimeError" in out
    assert len(calls.save_report) == 1
    assert calls.save_report[0]["tool_comparison"] is None
    print("[OK] A/B 失败降级：主报告不受影响")


# ---------------------------------------------------------------- 终端结论
def test_print_tool_ab_reports_key_metrics(capsys):
    """对比表结论必须给出延迟变化、token 倍数与自主检索比例三个关键指标。"""
    run_eval._print_tool_ab(_comparison())
    out = capsys.readouterr().out
    assert "延迟变化" in out and "-50.0%" in out
    assert "token 倍数" in out and "0.50x" in out
    assert "自主检索比例" in out and "50.0%" in out
    print("[OK] A/B 终端结论三个关键指标就位")
