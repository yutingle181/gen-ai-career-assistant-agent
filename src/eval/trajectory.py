"""轨迹级评测（P1-5）：不只看「答得对不对」，还看「是怎么走到的」。

为什么必须单独一层：`metrics.py` 的 Recall@K / MRR / 幻觉率评的是**一次回答**，
而 Agent 的失败常常不在最终文本里——它可能靠十几次盲目重试撞对答案，
也可能工具失败后硬撑着编了一版「看起来对」的结论。这类问题只有看**整条轨迹**
（轮次、工具调用、失败发生在第几步）才会暴露。

三个口径（全部写死，保证跨报告可比）：
1. **TaskSuccessRate**：按场景定义「成功」，不是笼统的「有输出」。
   判定只看确定性的结构标记与工具事件，不再调一次模型（省成本、可复现）：
   - 空回答 / 命中失败兜底文案 → 失败；
   - 场景要求的结构标记（如复盘的 `## 面试复盘`）缺失 → 失败；
   - `REQUIRE_TOOLS_OK` 里的场景（知识库 / 职位检索）若有工具降级 → 失败
     （没检索到外部资料还给出确定性结论，本身就是不可信的成功）。
2. **Failure Onset**：错误最早出现在第几步。平均值能区分「一上来就错」和
   「前面都对、最后一步崩」，这决定了修复该从哪一层下手。
3. **代价指标**：工具失败率、达到轮次上限的比例、平均轮次——
   用于回答「这个成功率花了多少代价换来的」。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: 各场景的「成功」所需结构标记（与各自的输出契约一一对应）。
SUCCESS_MARKERS: dict[str, tuple[str, ...]] = {
    "knowledge": ("[1]",),  # 知识库问答必须带引用编号
    "jd_match": ("## JD 匹配诊断",),
    "interview_review": ("## 面试复盘",),
    "job_search": ("| 公司 |",),  # 结构化清单（Markdown 表格）
    "mock_interview": (),  # 多轮对话：只要不是兜底失败即算成功
}

#: 这些场景的结论依赖外部资料：工具降级时即便文本工整，也不计成功。
REQUIRE_TOOLS_OK: frozenset[str] = frozenset({"knowledge", "job_search"})

#: 会话层失败兜底文案（命中即判失败）。刻意用前缀匹配，避免因措辞微调失效。
FAILURE_MARKERS: tuple[str, ...] = (
    "（模型调用失败",
    "（未配置 OPENAI_API_KEY",
    "（输入内容未通过安全检查",
)

#: 图级递归上限的兜底文案前缀（来自 graph/tool_loop，避免两处各写一份）。
RECURSION_FALLBACK_PREFIX = "（本轮工具调用已达执行上限"


@dataclass
class TrajectorySample:
    """一条轨迹样本：最终文本 + 工具事件 + 代价信息。"""

    mode: str = ""
    text: str = ""
    events: list[dict] = field(default_factory=list)
    rounds: int = 0
    latency_ms: int = 0
    tokens: int = 0
    error: str = ""


@dataclass
class TrajectoryStats:
    """轨迹级汇总指标。"""

    samples: int = 0
    success: int = 0
    success_rate: float = 0.0
    failed_reasons: dict[str, int] = field(default_factory=dict)
    tool_failure_rate: float = 0.0  # 有工具降级的样本占比
    over_step_rate: float = 0.0  # 撞到轮次上限 / 递归兜底的样本占比
    avg_rounds: float = 0.0
    failure_onset_hist: dict[int, int] = field(default_factory=dict)
    avg_failure_onset: float | None = None
    onset_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转成普通 dict（便于塞进报告快照 / 结构化返回）。"""
        return {
            "samples": self.samples,
            "success": self.success,
            "success_rate": self.success_rate,
            "failed_reasons": self.failed_reasons,
            "tool_failure_rate": self.tool_failure_rate,
            "over_step_rate": self.over_step_rate,
            "avg_rounds": self.avg_rounds,
            "failure_onset_hist": {str(k): v for k, v in self.failure_onset_hist.items()},
            "avg_failure_onset": self.avg_failure_onset,
            "onset_examples": self.onset_examples,
        }


def success_of(mode: str, text: str, events: Sequence[dict] | None = None) -> tuple[bool, str]:
    """按场景判定是否成功，返回 (是否成功, 判定依据)。

    判定依据一定要返回：失败样本要能回答「为什么算失败」，
    否则 TaskSuccessRate 只是一个没法行动的数字。
    """
    body = (text or "").strip()
    if not body:
        return False, "空回答"

    for marker in FAILURE_MARKERS:
        if marker in body:
            return False, "命中失败兜底文案"

    missing = [m for m in SUCCESS_MARKERS.get(mode, ()) if m not in body]
    if missing:
        return False, f"缺少场景结构标记：{'、'.join(missing)}"

    event_list = list(events or [])
    degraded = [e for e in event_list if not e.get("ok", True)]
    if degraded and mode in REQUIRE_TOOLS_OK:
        kinds = "、".join(sorted({str(e.get("error_kind") or "unknown") for e in degraded}))
        return False, f"依赖检索的场景出现工具降级（{kinds}）"

    return True, "符合场景成功口径"


def failure_onset(events: Sequence[dict]) -> int | None:
    """错误最早出现在第几步（从 1 计）；整条轨迹无失败步则返回 None。"""
    for index, event in enumerate(events, start=1):
        if not event.get("ok", True):
            return index
    return None


def reached_step_limit(sample: TrajectorySample, max_steps: int | None = None) -> bool:
    """是否撞到轮次上限（或触发了图级递归兜底）。

    两种信号都算：工具调用次数达到上限，或最终文本是递归兜底文案
    （后者说明连收束节点都没能正常走到）。
    """
    from .. import config

    limit = config.TOOL_CALLING_MAX_STEPS if max_steps is None else max_steps
    if (sample.text or "").lstrip().startswith(RECURSION_FALLBACK_PREFIX):
        return True
    return len(sample.events) >= max(1, limit)


def summarize(samples: Sequence[TrajectorySample], max_steps: int | None = None) -> TrajectoryStats:
    """把逐样本轨迹汇总为指标；空输入返回全 0 而不除零。"""
    total = len(samples)
    if not total:
        return TrajectoryStats()

    success = 0
    reasons: dict[str, int] = {}
    with_tool_failure = 0
    over_step = 0
    onsets: list[int] = []
    hist: dict[int, int] = {}
    examples: list[str] = []

    for sample in samples:
        ok, reason = success_of(sample.mode, sample.text, sample.events)
        if ok:
            success += 1
        else:
            # 归并原因（去掉具体标记名，保留可聚合的类别）
            key = reason.split("：")[0]
            reasons[key] = reasons.get(key, 0) + 1

        if any(not e.get("ok", True) for e in sample.events):
            with_tool_failure += 1
        if reached_step_limit(sample, max_steps):
            over_step += 1

        onset = failure_onset(sample.events)
        if onset is not None:
            onsets.append(onset)
            hist[onset] = hist.get(onset, 0) + 1
            if len(examples) < 3:
                examples.append(f"{sample.mode or 'unknown'} 第 {onset} 步：{self_reason(sample, onset)}")

    rounds = [s.rounds for s in samples if s.rounds > 0]
    return TrajectoryStats(
        samples=total,
        success=success,
        success_rate=round(success / total, 4),
        failed_reasons=reasons,
        tool_failure_rate=round(with_tool_failure / total, 4),
        over_step_rate=round(over_step / total, 4),
        avg_rounds=round(sum(rounds) / len(rounds), 2) if rounds else 0.0,
        failure_onset_hist=hist,
        avg_failure_onset=round(sum(onsets) / len(onsets), 2) if onsets else None,
        onset_examples=examples,
    )


def self_reason(sample: TrajectorySample, onset: int) -> str:
    """给出「第 N 步为什么算失败」的短说明（工具名 + 失败分类）。"""
    event = sample.events[onset - 1] if 0 < onset <= len(sample.events) else {}
    name = str(event.get("name") or "unknown")
    kind = str(event.get("error_kind") or "")
    return f"{name} 失败（{kind or '未分类'}）"


def stats_from_outcomes(
    outcomes: Sequence[Any], *, mode: str = "qa", max_steps: int | None = None
) -> TrajectoryStats:
    """从 `tool_eval.PathOutcome` 列表构造轨迹统计。

    刻意用鸭子类型取值（`getattr`）而不是 import `PathOutcome`：
    评测模块不该为了几个字段反向依赖任务图与执行器，否则以后换 runner 就要改这里。
    """
    samples = [
        TrajectorySample(
            mode=mode,
            text=getattr(o, "text", "") or "",
            events=list(getattr(o, "events", []) or []),
            rounds=int(getattr(o, "rounds", 0) or 0),
            latency_ms=int(getattr(o, "latency_ms", 0) or 0),
            tokens=int(getattr(o, "total_tokens", 0) or 0),
            error=getattr(o, "error", "") or "",
        )
        for o in outcomes
    ]
    return summarize(samples, max_steps=max_steps)


def render_trajectory(stats: dict[str, Any] | TrajectoryStats, index: int = 4) -> list[str]:
    """渲染轨迹评测章节（行列表，便于嵌进总报告）。"""
    data = stats.to_dict() if isinstance(stats, TrajectoryStats) else dict(stats or {})
    total = int(data.get("samples") or 0)
    lines: list[str] = [
        "",
        f"## {index}. 轨迹级评测（TaskSuccessRate / Failure Onset）",
        "",
        "口径：成功由场景结构标记与工具事件判定（不额外调用模型，可复现）；",
        "「依赖检索的场景」出现工具降级时即便文本工整也判为失败。",
        "",
    ]
    if not total:
        lines.append("- 无有效轨迹样本。")
        return lines

    success_rate = float(data.get("success_rate") or 0.0)
    lines += [
        f"- **TaskSuccessRate**：{success_rate * 100:.1f}%（{data.get('success')}/{total}）",
        f"- 工具失败率：{float(data.get('tool_failure_rate') or 0) * 100:.1f}%"
        f"（有工具降级的样本占比）",
        f"- 达到轮次上限的比例：{float(data.get('over_step_rate') or 0) * 100:.1f}%",
        f"- 平均轮次：{data.get('avg_rounds')}",
    ]
    onset = data.get("avg_failure_onset")
    lines.append(
        f"- **Failure Onset 均值**：{onset}（越小说明越早出错，越该先修前面的环节）"
        if onset is not None
        else "- **Failure Onset**：本批样本没有工具失败步。"
    )

    # 对照路径的降级必须单列：它是最常跑的生产路径，检索挂掉时
    # 只看实验组会得出「一切正常」的假结论（故障注入实验已撞到过这一点）。
    explicit = data.get("explicit_degradation") or {}
    if explicit.get("samples"):
        onset_ex = explicit.get("avg_failure_onset")
        tail = f"，Failure Onset 均值 {onset_ex}" if onset_ex is not None else "，无失败步"
        lines.append(
            f"- 对照组（显式检索，{explicit['samples']} 条）：工具失败率 "
            f"{float(explicit.get('tool_failure_rate') or 0) * 100:.1f}%{tail}"
        )

    reasons = data.get("failed_reasons") or {}
    if reasons:
        lines += ["", "失败原因分布：", ""]
        lines += [f"- {k}：{v} 条" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])]

    hist = data.get("failure_onset_hist") or {}
    if hist:
        lines += ["", "Failure Onset 直方图（第几步）：", ""]
        lines += [
            f"- 第 {step} 步：{count} 条"
            for step, count in sorted(hist.items(), key=lambda kv: int(kv[0]))
        ]

    examples = data.get("onset_examples") or []
    if examples:
        lines += ["", "最早失败示例：", ""]
        lines += [f"- {e}" for e in examples]
    return lines
