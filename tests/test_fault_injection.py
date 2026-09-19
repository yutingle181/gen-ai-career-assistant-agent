"""故障注入：把「失败路径」变成可复现、可断言的用例。

为什么值得单独一个文件：失败分级（P0-2）、熔断（P2-7）、Failure Onset（P1-5）
这三块能力在 happy path 上永远是「0 失败、100% 成功」——那样的报告什么都证明不了。
这里用固定种子制造可控故障，断言「上游坏掉时到底会发生什么」。

全部离线：不联网、不需要 API Key、不真的等待超时。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import config, faults, metrics, tools


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """默认关闭注入；用例需要时显式打开，结束后状态清干净。"""
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", False)
    monkeypatch.setattr(config, "FAULT_INJECT_TOOL", "knowledge_search")
    monkeypatch.setattr(config, "FAULT_INJECT_KIND", "http_5xx")
    monkeypatch.setattr(config, "FAULT_INJECT_RATE", 1.0)
    monkeypatch.setattr(config, "FAULT_INJECT_SEED", 42)
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)
    monkeypatch.setattr(config, "WEB_SEARCH_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(config, "WEB_SEARCH_RETRY_BACKOFF", 0.0)
    faults.reset()
    tools.get_breaker().reset()
    metrics.reset()
    yield
    faults.reset()
    tools.get_breaker().reset()
    metrics.reset()


def _kb_tool(monkeypatch, chunks: list | None = None):
    """构造一个把 registry 打桩掉的知识库工具（避免真建库 / 真检索）。"""

    class _Pipeline:
        def retrieve(self, query, cfg=None):
            return list(chunks or [])

    class _Registry:
        def get(self, name):
            return _Pipeline()

    import src.rag.registry as registry

    monkeypatch.setattr(registry, "get_registry", lambda: _Registry())
    return tools._knowledge_search_tool("kb")


def _chunk(source: str = "s.md", page: int = 1, text: str = "片段内容"):
    return SimpleNamespace(source=source, page=page, text=text, meta={})


# ------------------------------------------------------------------ 注入口本身
def test_injection_off_by_default():
    """默认关闭：所有注入口都是无副作用的空操作（零行为变化）。"""
    assert faults.is_enabled() is False
    assert faults.maybe_fail("knowledge_search") is None
    assert faults.maybe_empty("knowledge_search") is False
    assert faults.describe()["enabled"] is False


def test_injection_targets_only_configured_tool(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    assert faults.maybe_fail("other_tool") is None
    assert faults.maybe_fail("knowledge_search") is not None


def test_simulated_errors_are_classified_like_real_ones(monkeypatch):
    """模拟故障必须能被真实的分级逻辑识别——否则这套「演示」证明不了任何东西。"""
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    for kind, expect in (("http_5xx", "http_5xx"), ("http_4xx", "http_4xx"), ("timeout", "timeout")):
        monkeypatch.setattr(config, "FAULT_INJECT_KIND", kind)
        faults.reset()
        exc = faults.maybe_fail("knowledge_search")
        assert exc is not None
        assert tools.classify_error(exc) == expect


def test_fault_sequence_is_reproducible(monkeypatch):
    """固定种子 + 固定调用顺序 ⇒ 同一串故障（报告可复现的前提）。"""
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    monkeypatch.setattr(config, "FAULT_INJECT_RATE", 0.5)

    def sequence() -> list[bool]:
        faults.reset()
        return [faults.maybe_fail("knowledge_search") is not None for _ in range(12)]

    first = sequence()
    assert first == sequence()
    assert any(first) and not all(first)  # 0.5 概率下应当有成功也有失败


def test_empty_kind_is_not_an_exception(monkeypatch):
    """empty 与「报错」语义不同：一个是「查了没结果」，一个是「没查成」。"""
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    monkeypatch.setattr(config, "FAULT_INJECT_KIND", "empty")

    assert faults.maybe_fail("knowledge_search") is None
    assert faults.maybe_empty("knowledge_search") is True


# ------------------------------------------------------------------ 知识库工具的失败分级
def test_knowledge_failure_is_graded_not_silently_empty(monkeypatch):
    """注入 5xx 后必须回「工具降级 + 分类」，而不是「未检索到相关内容」。

    这正是本次修掉的语义漏洞：前者让模型知道「没查成」，后者会让它以为「知识库里没有」——
    后者会诱使模型理直气壮地给出没有依据的结论。
    """
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    tool = _kb_tool(monkeypatch)

    out = tool.invoke({"query": "问题"})

    assert tools.degraded_kind(out) == "http_5xx"
    assert "未检索到相关内容" not in out
    # 免责声明必须指向「知识库检索」，不能说成「未联网」——否则排查方向会被带偏
    assert "知识库检索暂不可用" in out and "未取得资料依据" in out
    assert metrics.counter("tool.fail.http_5xx") == 1
    assert metrics.counter("tool.call.fail") == 1


def test_knowledge_retries_transient_then_succeeds(monkeypatch):
    """瞬态故障会重试：失败两次后第三次成功（与联网检索同一套口径）。"""
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    monkeypatch.setattr(config, "WEB_SEARCH_RETRY_ATTEMPTS", 3)
    rolls = {"n": 0}

    def _fail_twice(tool_name):
        rolls["n"] += 1
        return TimeoutError("simulated timeout") if rolls["n"] <= 2 else None

    monkeypatch.setattr(faults, "maybe_fail", _fail_twice)
    tool = _kb_tool(monkeypatch, chunks=[_chunk()])

    out = tool.invoke({"query": "问题"})

    assert rolls["n"] == 3  # 两次失败 + 第三次成功
    assert tools.degraded_kind(out) == ""
    assert "片段内容" in out


def test_knowledge_failure_opens_breaker_and_short_circuits(monkeypatch):
    """连续失败到阈值后熔断：后续调用直接短路，不再打上游。"""
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    monkeypatch.setattr(config, "TOOL_BREAKER_THRESHOLD", 2)
    tool = _kb_tool(monkeypatch)

    assert tools.degraded_kind(tool.invoke({"query": "a"})) == "http_5xx"
    assert tools.degraded_kind(tool.invoke({"query": "b"})) == "http_5xx"
    assert tools.degraded_kind(tool.invoke({"query": "c"})) == "circuit_open"

    assert tools.get_breaker().snapshot()["open"] == ["knowledge_search"]
    assert metrics.counter("tool.call.blocked") == 1


def test_web_search_failure_is_degraded_with_kind(monkeypatch):
    """联网检索路径同样受注入口驱动（两条工具走同一套契约）。

    尝试次数 = 策略数 × 每通道重试数：对短问题而言策略是「原式」+「备选通道」两条，
    每条重试 2 次 ⇒ 4 次。这个数字正好把「重试 → 换通道」整个阶梯暴露出来，
    而不是只重试同一个通道。
    """
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    monkeypatch.setattr(config, "FAULT_INJECT_TOOL", "search_web")
    monkeypatch.setattr(config, "WEB_SEARCH_RETRY_ATTEMPTS", 2)

    outcome = tools.web_search_detailed("任意问题")

    assert outcome.error_kind == "http_5xx"
    assert len(tools._search_plan("任意问题")) == 2  # 原式 + 备选通道
    assert outcome.attempts == 4
    assert metrics.counter("tool.fail.http_5xx") == 1


# ------------------------------------------------------------------ 统一检索契约
def test_retrieve_with_grade_degrades_instead_of_raising(monkeypatch):
    """检索失败返回 (False, [], kind)，绝不抛异常——调用方无处可逃地要处理它。"""
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    calls = {"n": 0}

    class _Pipeline:
        def retrieve(self, query, cfg=None, use_cache=True):
            calls["n"] += 1
            raise AssertionError("注入口应先抛错，不该真的走到 pipeline")

    assert tools.retrieve_with_grade(_Pipeline(), "问题") == (False, [], "http_5xx")
    assert calls["n"] == 0


def test_retrieve_with_grade_opens_breaker(monkeypatch):
    """同一份熔断计数覆盖工具路径与显式检索路径（共用工具名 knowledge_search）。"""
    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    monkeypatch.setattr(config, "TOOL_BREAKER_THRESHOLD", 2)

    class _Pipeline:
        def retrieve(self, query, cfg=None, use_cache=True):
            return []

    kinds = [tools.retrieve_with_grade(_Pipeline(), f"q{i}")[2] for i in range(3)]

    assert kinds == ["http_5xx", "http_5xx", "circuit_open"]
    assert tools.get_breaker().snapshot()["open"] == ["knowledge_search"]
    assert metrics.counter("tool.call.blocked") == 1


def test_explicit_retrieval_path_degrades_not_crashes(monkeypatch):
    """对照（显式检索）路径：检索挂掉时不再崩，而是把降级说明交给模型。

    这是本次故障注入暴露出的真实缺口：原实现直接 `pipeline.retrieve()`，
    上游一抛错整条样本就崩，或静默拿到空上下文 —— 模型会把「没查成」当成
    「知识库里没有」，然后凭记忆给出没有依据的结论。
    """
    from src.eval.tool_eval import make_explicit_runner

    monkeypatch.setattr(config, "ENABLE_FAULT_INJECTION", True)
    monkeypatch.setattr(config, "WEB_SEARCH_RETRY_ATTEMPTS", 1)

    class _Pipeline:
        def retrieve(self, query, cfg=None, use_cache=True):
            raise AssertionError("注入口应先抛错")

    sent: dict = {}

    def _fake_invoke(messages, model=None):
        sent["prompt"] = messages[-1].content
        return "抱歉，本轮未取得资料依据，以下为未核实的推测。"

    outcome = make_explicit_runner(_Pipeline(), llm=_fake_invoke)(
        SimpleNamespace(question="问题")
    )

    assert outcome.ok is True  # 不再抛异常
    assert outcome.events[0] == {
        "name": "knowledge_search",
        "ok": False,
        "error_kind": "http_5xx",
    }
    assert "知识库检索暂不可用" in sent["prompt"]
    assert outcome.citations == []


# ------------------------------------------------------------------ 轨迹指标在故障下才有意义
def test_trajectory_metrics_catch_injected_failure(monkeypatch):
    """有故障时 TaskSuccessRate / Failure Onset 才会动 —— 说明它们不是摆设。"""
    from src.eval.trajectory import failure_onset, stats_from_outcomes, success_of

    sample = SimpleNamespace(
        text="结论来自资料 [1]",
        events=[{"name": "knowledge_search", "ok": False, "error_kind": "http_5xx"}],
        rounds=2,
    )
    stats = stats_from_outcomes([sample], mode="knowledge", max_steps=3)

    assert stats.success_rate == 0.0
    assert stats.tool_failure_rate == 1.0
    assert stats.avg_failure_onset == 1.0
    assert failure_onset(sample.events) == 1

    ok, reason = success_of("knowledge", sample.text, sample.events)
    assert ok is False and "工具降级" in reason


# ------------------------------------------------------------------ 故障必须出现在报告里
def test_report_surfaces_explicit_path_degradation():
    """对照（显式检索）路径的降级要单列，不能显示成「一切正常」。

    这是本次故障注入暴露的第二个问题：15/15 次检索全挂，报告却写「工具失败率 0%、
    无失败步」——因为轨迹只统计实验组，而实验组这轮一次工具都没调。
    """
    from src.eval.tool_eval import PathOutcome, _trajectory_block
    from src.eval.trajectory import render_trajectory

    degraded_event = {"name": "knowledge_search", "ok": False, "error_kind": "http_5xx"}
    explicit = [PathOutcome(ok=True, text="x", events=[degraded_event]) for _ in range(3)]
    tool = [PathOutcome(ok=True, text="x", events=[]) for _ in range(3)]

    block = _trajectory_block(explicit, tool)

    assert block["tool_failure_rate"] == 0.0  # 实验组确实没有工具失败
    assert block["explicit_degradation"]["tool_failure_rate"] == 1.0
    assert block["explicit_degradation"]["avg_failure_onset"] == 1.0

    lines = render_trajectory(block)
    assert any("对照组（显式检索，3 条）" in line and "100.0%" in line for line in lines)


def test_compare_paths_wires_explicit_degradation():
    """接线检查：compare_paths 产出的轨迹块里必须带上对照组降级。"""
    from src.eval.tool_eval import PathOutcome, compare_paths

    record = SimpleNamespace(question="q")
    degraded_event = {"name": "knowledge_search", "ok": False, "error_kind": "http_5xx"}

    def _explicit(_record):
        return PathOutcome(ok=True, text="a", tool_calls=1, rounds=1, events=[degraded_event])

    def _tool(_record):
        return PathOutcome(ok=True, text="b", tool_calls=0, rounds=1, events=[])

    comparison = compare_paths([record, record], _explicit, _tool, max_workers=1)

    assert comparison.trajectory["explicit_degradation"]["tool_failure_rate"] == 1.0
    assert comparison.trajectory["explicit_degradation"]["samples"] == 2


# ------------------------------------------------------------------ 评测 CLI 契约
def test_eval_cli_exposes_fault_flags():
    """注入参数必须能只靠命令行复现（报告要可复现，命令就得写全）。"""
    from run_eval import build_parser

    default = build_parser().parse_args([])
    assert default.fault_tool == "" and default.fault_rate == 0.0
    assert default.skip_rag is False and default.fault_kind == "http_5xx"

    args = build_parser().parse_args(
        ["--fault-tool", "knowledge_search", "--fault-kind", "timeout",
         "--fault-rate", "0.3", "--fault-seed", "7", "--skip-rag"]
    )
    assert (args.fault_tool, args.fault_kind, args.fault_rate, args.fault_seed) == (
        "knowledge_search",
        "timeout",
        0.3,
        7,
    )
    assert args.skip_rag is True


def test_apply_fault_injection_sets_config():
    """CLI 参数 → config 的落点要正确；不填 rate 视为 1.0（全失败）。"""
    from run_eval import apply_fault_injection, build_parser

    apply_fault_injection(
        build_parser().parse_args(["--fault-tool", "knowledge_search", "--fault-kind", "timeout"])
    )

    assert config.ENABLE_FAULT_INJECTION is True
    assert config.FAULT_INJECT_TOOL == "knowledge_search"
    assert config.FAULT_INJECT_KIND == "timeout"
    assert config.FAULT_INJECT_RATE == 1.0
    assert faults.is_enabled() is True


def test_apply_fault_injection_is_noop_without_flag():
    from run_eval import apply_fault_injection, build_parser

    apply_fault_injection(build_parser().parse_args([]))
    assert config.ENABLE_FAULT_INJECTION is False
    assert faults.is_enabled() is False
