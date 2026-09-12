"""显式检索路径 vs Function Calling 路径的 A/B 评测。

为什么单独一份：`runner.py` 衡量的是**检索质量**（Recall@K / MRR / 幻觉率），
而「要不要让模型自主调用工具」是一个**系统级取舍**——它不改召回算法，改的是
延迟、token 成本、模型往返轮次与最终成功率。两类指标混进一张表会互相污染结论，
因此这里独立成 `ToolPathMetric` 并单独成章。

成本口径（务必诚实）：兼容接口不总是回传真实用量，这里统一用 `tiktoken`
对「提示文本 + 回答文本」做**近似计数**，用于比较两条路径的**相对**成本，
不等于计费口径的绝对 token 数。

并发与缓存约定沿用 `runner.py`（`_resolve_workers` / `_map_concurrent`），
检索一律 `use_cache=False`，避免预热后延迟被缓存打平。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .. import config

# token 计数复用 cache.py 既有实现（tiktoken + 中文折算回退），避免第二套口径导致成本数字打架
from ..cache import count_tokens
from ..logging_setup import get_logger
from ..state import MODE_QA
from .runner import _map_concurrent, _resolve_workers

logger = get_logger(__name__)

PATH_EXPLICIT = "显式检索"
PATH_TOOL_CALLING = "Function Calling"

# 单条样本的执行结果（由各路径的 runner 产出）
PathRunner = Callable[[object], "PathOutcome"]


@dataclass
class PathOutcome:
    """单条样本在一条路径上的执行结果。"""

    ok: bool = True
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rounds: int = 1  # 模型往返轮次（显式路径固定 1 轮）
    tool_calls: int = 0  # 实际执行的工具调用次数
    citations: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ToolPathMetric(BaseModel):
    """一条路径的汇总指标（对比表的一行）。"""

    name: str
    samples: int = 0
    failures: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    total_tokens: int = 0
    avg_rounds: float = 0.0
    avg_tool_calls: float = 0.0
    tool_call_rate: float = 0.0  # 主动调用过工具的样本占比
    config: dict = Field(default_factory=dict)


class ToolComparison(BaseModel):
    """双路径对比结果。"""

    explicit: ToolPathMetric
    tool_calling: ToolPathMetric
    sample_count: int = 0

    def rows(self) -> list[tuple[str, ToolPathMetric]]:
        return [(PATH_EXPLICIT, self.explicit), (PATH_TOOL_CALLING, self.tool_calling)]

    def delta(self, key: str) -> float:
        """实验组相对对照组的变化（绝对值）。"""
        base = getattr(self.explicit, key, 0.0)
        new = getattr(self.tool_calling, key, 0.0)
        return float(new) - float(base)


# ---------------------------------------------------------------- 汇总
def _p95(values: Sequence[int]) -> float:
    """第 95 百分位（最近秩法），用于观察长尾而非只看均值。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return float(ordered[idx])


def summarize_path(
    name: str,
    outcomes: Sequence[PathOutcome],
    *,
    config_snapshot: dict | None = None,
) -> ToolPathMetric:
    """把逐样本结果汇总为一行对比指标；空输入返回全 0 而不除零。"""
    total = len(outcomes)
    if not total:
        return ToolPathMetric(name=name, config=config_snapshot or {})

    ok_outs = [o for o in outcomes if o.ok]
    latencies = [o.latency_ms for o in ok_outs]
    tokens = [o.total_tokens for o in ok_outs]
    tool_calls = [o.tool_calls for o in ok_outs]
    rounds = [o.rounds for o in ok_outs]

    def _avg(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    return ToolPathMetric(
        name=name,
        samples=total,
        failures=total - len(ok_outs),
        success_rate=round(len(ok_outs) / total, 4),
        avg_latency_ms=_avg(latencies),
        p95_latency_ms=_p95(latencies),
        avg_tokens=_avg(tokens),
        total_tokens=sum(tokens),
        avg_rounds=_avg(rounds),
        avg_tool_calls=_avg(tool_calls),
        tool_call_rate=round(sum(1 for c in tool_calls if c > 0) / len(ok_outs), 4) if ok_outs else 0.0,
        config=config_snapshot or {},
    )


def _safe_run(runner: PathRunner, record) -> PathOutcome:
    """执行单条样本；任何异常都记为失败样本，不中断整批评测。"""
    try:
        outcome = runner(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("路径执行失败，计为失败样本：%s", exc)
        return PathOutcome(ok=False)
    return outcome if isinstance(outcome, PathOutcome) else PathOutcome(ok=False)


def compare_paths(
    records: Sequence[object],
    explicit_runner: PathRunner,
    tool_runner: PathRunner,
    *,
    max_workers: int | None = None,
    explicit_config: dict | None = None,
    tool_config: dict | None = None,
) -> ToolComparison:
    """在同一批样本上并发跑两条路径并汇总对比。

    两条路径各自独立并发：显式路径不依赖工具调用路径的状态，
    分开跑可以避免其中一条的慢网络拖住另一条的吞吐。
    """
    records = list(records)
    workers = _resolve_workers(max_workers, len(records))

    explicit_outs = _map_concurrent(lambda r: _safe_run(explicit_runner, r), records, workers)
    tool_outs = _map_concurrent(lambda r: _safe_run(tool_runner, r), records, workers)

    comparison = ToolComparison(
        explicit=summarize_path(PATH_EXPLICIT, explicit_outs, config_snapshot=explicit_config),
        tool_calling=summarize_path(PATH_TOOL_CALLING, tool_outs, config_snapshot=tool_config),
        sample_count=len(records),
    )
    logger.info(
        "双路径对比完成 | 样本=%d | 显式 %.0fms/%.0ftoken | 工具调用 %.0fms/%.0ftoken | 工具调用率 %.0f%%",
        len(records),
        comparison.explicit.avg_latency_ms,
        comparison.explicit.avg_tokens,
        comparison.tool_calling.avg_latency_ms,
        comparison.tool_calling.avg_tokens,
        comparison.tool_calling.tool_call_rate * 100,
    )
    return comparison


# ---------------------------------------------------------------- 真实路径
def make_explicit_runner(
    pipeline,
    *,
    retrieval_cfg=None,
    persona: str | None = None,
    llm=None,
    model: str | None = None,
) -> PathRunner:
    """构造对照组 runner：固定先检索一次，再把结果作为上下文交给模型作答。

    ``model`` 用于把两条路径钉在同一个模型上做公平对比（默认走 config.MODEL_NAME）。
    """

    def _run(record) -> PathOutcome:
        from langchain_core.messages import HumanMessage, SystemMessage

        from ..llm import llm_invoke
        from ..prompts.personas import get_persona

        system = persona or get_persona(MODE_QA)
        question = getattr(record, "question", str(record))
        invoke = llm or llm_invoke

        start = time.perf_counter()
        chunks = pipeline.retrieve(question, retrieval_cfg, use_cache=False)
        context = "\n\n".join(getattr(c, "text", "") for c in chunks)
        prompt = f"{question}\n\n【检索结果】\n{context}"
        answer = invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)], model=model
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        return PathOutcome(
            ok=True,
            latency_ms=latency_ms,
            prompt_tokens=count_tokens(system) + count_tokens(prompt),
            completion_tokens=count_tokens(answer or ""),
            rounds=1,
            tool_calls=1,  # 显式路径的「一次检索」即其工具调用
            citations=[getattr(c, "chunk_id", "") for c in chunks],
        )

    return _run


def make_tool_runner(
    tools: list,
    *,
    persona: str | None = None,
    max_steps: int | None = None,
    runner=None,
    model: str | None = None,
) -> PathRunner:
    """构造实验组 runner：让模型自主决定是否/如何调用工具（ReAct 闭环）。

    ``model`` 语义同显式路径：两条路径必须用同一模型，否则对比的是模型而非路径。
    """

    def _run(record) -> PathOutcome:
        from langchain_core.messages import HumanMessage, SystemMessage

        from ..graph.tool_loop import run_with_tools
        from ..prompts.personas import get_persona

        system = persona or get_persona(MODE_QA)
        question = getattr(record, "question", str(record))
        events: list[dict] = []
        call = runner or run_with_tools

        start = time.perf_counter()
        answer = call(
            [SystemMessage(content=system), HumanMessage(content=question)],
            tools=tools,
            model=model,
            max_steps=max_steps,
            on_event=events.append,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        return PathOutcome(
            ok=True,
            latency_ms=latency_ms,
            prompt_tokens=count_tokens(system) + count_tokens(question),
            completion_tokens=count_tokens(answer or ""),
            rounds=1 + len(events),  # 首轮决策 + 每次工具回灌后各一轮
            tool_calls=len(events),
        )

    return _run


def run_tool_ab(
    pipeline,
    records: Sequence[object],
    *,
    kb_name: str | None = None,
    retrieval_cfg=None,
    max_workers: int | None = None,
    model: str | None = None,
    include_web: bool = True,
) -> ToolComparison:
    """端到端跑一次 A/B（需要可用 API Key 与已建好的知识库）。

    ``model`` 建议显式指定，确保两条路径同模型；报告里会一并记录，便于复现。
    """
    from ..tools import get_agent_tools

    # 同一套检索配置交给两条路径，确保对比的是「链路」而不是「检索参数」
    tools = get_agent_tools(kb_name, include_web=include_web, retrieval_cfg=retrieval_cfg)
    used_model = model or config.MODEL_NAME
    return compare_paths(
        records,
        make_explicit_runner(pipeline, retrieval_cfg=retrieval_cfg, model=model),
        make_tool_runner(tools, model=model),
        max_workers=max_workers,
        explicit_config={
            "path": "explicit",
            "model": used_model,
            "retrieval": getattr(retrieval_cfg, "model_dump", lambda: {})(),
        },
        tool_config={
            "path": "tool_calling",
            "model": used_model,
            "max_steps": config.TOOL_CALLING_MAX_STEPS,
            "tools": [getattr(t, "name", str(t)) for t in tools],
        },
    )
