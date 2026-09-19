"""工具层：联网搜索（可降级 + 失败分级）+ 供 Function Calling 使用的工具定义。

对应 JD 要求 3「Function Calling」：本模块同时提供
- `web_search()`：显式检索（默认路径，稳定、低延迟）；
- `get_agent_tools()`：bind_tools 工具调用版本（由 `ENABLE_TOOL_CALLING` 开关启用）。

两者并存本身就是一个可讲的技术决策。关键约定：两条路径**共享同一套**
「守护线程超时 + 失败分级 + 降级」实现——此前工具列表里直接放入裸
`DuckDuckGoSearchResults`，那条路径没有超时保护，与显式检索路径行为不一致，
弱网下会把整轮对话拖住。

生产级补强（P0-2 / P0-3）：
1. **失败分级**：把「怎么坏的」结构化成 `ToolOutcome.error_kind`（超时 / 网络 / 5xx /
   4xx / 无结果），据此决定「重试 → 换参数（简化检索式）→ 换通道（备选 backend）」，
   并把失败原因**如实**回灌给模型（带降级标记），而不是压成一句含糊的「暂不可用」。
   为什么重要：模型不知道工具坏了就会继续装作查过——「工具失败」必须成为一种
   它能读到、能写进答案里的事实。
2. **数据时效**：检索结果注入抓取时间，供 Prompt 判断资料新鲜度
   （配合 `prompts/personas.py` 的时效声明规则）。
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from . import config, metrics
from .logging_setup import get_logger, summarize

logger = get_logger(__name__)

_T = TypeVar("_T")

# ---------------------------------------------------------------- 失败分级
#: 失败分类 → 中文标签。标签会出现在日志、回灌给模型的提示、以及事件流里，
#: 因此这里刻意用「人能读懂」的措辞，而不是直接抛 code。
ERROR_LABELS: dict[str, str] = {
    "timeout": "超时",
    "network": "网络不可达",
    "http_5xx": "上游服务不可用",
    "http_4xx": "请求被拒绝",
    "empty": "无结果",
    "disabled": "功能已关闭",
    "circuit_open": "熔断中",
    "unknown": "未知错误",
}

#: 参与熔断统计的工具名（与 bind_tools 暴露的名字一致，便于对照事件流）。
_SEARCH_TOOL_NAME = "search_web"

#: 值得重试的分类：都是**瞬态**故障。
#: 刻意排除 http_4xx（参数/权限问题，重试只会浪费一次超时）与 disabled。
RETRYABLE_ERRORS: frozenset[str] = frozenset({"timeout", "network", "http_5xx", "unknown"})

#: 降级标记前缀，形如 `【工具降级:timeout】`。
#: 存在的意义：工具是「软失败」（返回一句说明而不是抛异常），上层需要一种
#: 稳定可解析的方式识别它 —— 事件流据此把 ok 标成 False，指标据此统计失败分类。
DEGRADED_MARK = "【工具降级"
_DEGRADED_RE = re.compile(r"^【工具降级:([a-z0-9_]+)】")


@dataclass
class ToolOutcome:
    """一次工具调用的结果与失败信息（供上层决策与透出）。"""

    text: str = ""
    error_kind: str = ""
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error_kind

    @property
    def label(self) -> str:
        return ERROR_LABELS.get(self.error_kind, self.error_kind)


def classify_error(exc: BaseException) -> str:
    """把异常归类为可决策的错误类型（只归类，不处理）。

    顺序有讲究：先看结构化信息（HTTP 状态码），再退化为文本匹配 ——
    不同 SDK 抛出的异常类型差异很大，文本兜底能覆盖绝大多数情况。
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return "http_5xx" if status >= 500 else "http_4xx"

    if isinstance(exc, TimeoutError):
        return "timeout"
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if any(code in msg for code in ("500", "502", "503", "504")):
        return "http_5xx"
    if any(code in msg for code in ("429", "401", "403", "400")):
        return "http_4xx"
    if any(kw in msg for kw in ("connection", "connect", "network", "resolve", "unreachable", "ssl")):
        return "network"
    return "unknown"


def degraded_kind(text: str) -> str:
    """从回灌文本中解析降级分类；不是降级文本则返回空串。"""
    match = _DEGRADED_RE.match(text or "")
    return match.group(1) if match else ""


def _degraded_text(error_kind: str) -> str:
    """把失败分类翻译成一句「事实说明」回灌给模型。

    措辞要点：说明**没拿到外部资料**、要求**明说未联网核实**，并保留旧文案里的
    「暂不可用」表述（既有调用方与用例依赖这一措辞，且它足够准确）。
    """
    label = ERROR_LABELS.get(error_kind, ERROR_LABELS["unknown"])
    if error_kind == "empty":
        sentence = "联网检索没有返回公开结果，请基于已有信息作答，并说明未检索到外部资料。"
    else:
        sentence = (
            f"联网检索暂不可用（{label}），本轮没有获得任何外部资料，"
            "请基于已有信息作答，并在回答中明确说明未联网核实。"
        )
    return f"{DEGRADED_MARK}:{error_kind}】{sentence}"


# ---------------------------------------------------------------- 超时执行
def _run_with_status(
    fn: Callable[[], _T], timeout: int, what: str
) -> tuple[bool, _T | None, str]:
    """在守护线程中执行 `fn`，返回 (是否成功, 结果, 失败分类)。

    与 `_run_with_timeout` 是同一实现的两种视图：那个只回答「有没有结果」，
    这个额外回答「是怎么坏的」——失败分级必须建立在能拿到异常类型之上。
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["r"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["err"] = exc
            logger.warning("%s失败，降级 | %s | %s", what, type(exc).__name__, exc)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.warning("%s超时（%ss），降级", what, timeout)
        return False, None, "timeout"
    if "err" in box:
        return False, None, classify_error(box["err"])
    return True, box.get("r"), ""


def _run_with_timeout(fn: Callable[[], _T], timeout: int, default: _T, what: str) -> _T:
    """在守护线程中执行 `fn`：超时或抛异常一律返回 `default`，绝不阻塞主流程。

    工具调用链路里「慢」比「错」更伤体验——底层客户端在弱网下可能长时间挂起，
    因此所有外部工具统一走这一实现，保证「同一件事只有一种行为」。
    """
    ok, value, _kind = _run_with_status(fn, timeout, what)
    return value if (ok and value is not None) else default


# ---------------------------------------------------------------- 熔断（P2-7）
class CircuitBreaker:
    """按工具维度的熔断器。

    为什么需要它：重试解决的是「单次抖动」，不解决「上游整体挂了」。
    没有熔断时，弱网环境下的每一次提问都要把重试次数走满——
    用户等到的是成倍的延迟，而我们什么都没拿到，还可能把上游继续打下去。

    行为：连续失败到阈值即打开，冷却期内直接短路（不发请求）；
    冷却结束后自动「半开」——放一次真实调用过去，成功即复位。
    """

    def __init__(self) -> None:
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def _threshold(self) -> int:
        return max(1, config.TOOL_BREAKER_THRESHOLD)

    def is_open(self, name: str) -> bool:
        """该工具当前是否处于熔断冷却期。"""
        if not config.ENABLE_TOOL_CIRCUIT_BREAKER:
            return False
        with self._lock:
            until = self._open_until.get(name, 0.0)
            if until and time.monotonic() < until:
                return True
            if until:  # 冷却结束：半开，允许下一次真实调用
                self._open_until.pop(name, None)
                self._failures[name] = 0
            return False

    def record(self, name: str, ok: bool) -> None:
        """记录一次真实调用的结果（成功即清零，失败累积到阈值则打开）。"""
        with self._lock:
            if ok:
                self._failures[name] = 0
                self._open_until.pop(name, None)
                return
            self._failures[name] = self._failures.get(name, 0) + 1
            if self._failures[name] >= self._threshold():
                cooldown = max(1, config.TOOL_BREAKER_COOLDOWN)
                self._open_until[name] = time.monotonic() + cooldown
                logger.warning(
                    "工具 %s 连续失败 %d 次，熔断 %ds（冷却期内直接短路）",
                    name,
                    self._failures[name],
                    cooldown,
                )

    def snapshot(self) -> dict[str, Any]:
        """当前熔断状态（供 /health 展示，用于解释「为什么这次没检索」）。"""
        now = time.monotonic()
        with self._lock:
            return {
                "open": sorted(n for n, until in self._open_until.items() if until > now),
                "failures": {n: c for n, c in self._failures.items() if c},
            }

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()
            self._open_until.clear()


_BREAKER = CircuitBreaker()


def get_breaker() -> CircuitBreaker:
    """获取全局熔断器（测试可重置）。"""
    return _BREAKER


# ---------------------------------------------------------------- 联网检索
def _simplify_query(query: str, limit: int) -> str:
    """「换参数」策略：去掉引号 / 书名号等噪声并截断，得到更适合检索的短查询。

    依据：检索式过长、引号过多是「返回空结果」的常见原因；先降复杂度再重试，
    比在同一串关键词上反复打同一接口更可能拿到结果。
    """
    cleaned = str(query or "")
    for ch in ('"', "'", "“", "”", "‘", "’", "【", "】", "《", "》", "(", ")", "（）", "\n", "\t"):
        cleaned = cleaned.replace(ch, " ")
    cleaned = " ".join(cleaned.split())
    return cleaned[: max(1, limit)].strip()


def _search_once(backend: str, query: str, max_results: int) -> str:
    """单次检索（不重试）。backend 为空或 auto 时使用默认通道。"""
    from langchain_community.tools import DuckDuckGoSearchResults

    kwargs: dict[str, Any] = {"num_results": max_results}
    if backend and backend != "auto":
        kwargs["backend"] = backend
    try:
        tool = DuckDuckGoSearchResults(**kwargs)
    except TypeError:
        # 兼容不支持 backend 参数的旧版 langchain_community：
        # 退回默认通道，而不是因为「换通道」这一个策略不可用就整体失败。
        tool = DuckDuckGoSearchResults(num_results=max_results)
    return tool.run(query) or ""


def _search_plan(query: str) -> list[tuple[str, str]]:
    """构造检索计划：原式 → 简化式 → 备选通道。

    顺序本身就是失败分级策略：同通道先原样重试（应对瞬态抖动），
    再换更宽松的参数，最后才换通道（不同 backend 走不同上游接口）。
    """
    plan: list[tuple[str, str]] = [("auto", query)]
    simplified = _simplify_query(query, config.WEB_SEARCH_SIMPLIFY_CHARS)
    if simplified and simplified != query:
        plan.append(("auto", simplified))
    alt = (config.WEB_SEARCH_ALT_BACKEND or "").strip()
    if alt and alt != "auto":
        plan.append((alt, simplified or query))
    return plan


def fetch_time_prefix(now: datetime | None = None) -> str:
    """抓取时间前缀（P0-3 数据时效）。

    模型本身没有时钟，「资料是什么时候抓的」只能由工具如实告知，
    否则一份三年前的结果会以「当前信息」的口吻被写进答案。
    """
    if not config.ENABLE_FETCH_TIME:
        return ""
    return f"【抓取时间：{(now or datetime.now()).strftime('%Y-%m-%d %H:%M')}】"


def web_search_detailed(
    query: str, max_results: int = 5, timeout: int | None = None
) -> ToolOutcome:
    """联网检索（分级重试版），返回结构化结果。

    失败处理顺序：同通道退避重试 → 简化检索式重试 → 换通道重试 → 如实降级。
    任何路径都不抛异常：工具层的「慢」和「坏」都不该拖垮对话，只该被如实记录与告知。
    """
    metrics.incr("tool.call.total")

    if not config.ENABLE_WEB_SEARCH:
        logger.info("联网搜索已关闭 | %s", summarize(query))
        return ToolOutcome(error_kind="disabled")

    breaker = get_breaker()
    if breaker.is_open(_SEARCH_TOOL_NAME):
        metrics.incr("tool.call.blocked")
        logger.warning("联网检索处于熔断冷却期，直接短路 | %s", summarize(query))
        return ToolOutcome(error_kind="circuit_open")

    timeout = config.WEB_SEARCH_TIMEOUT if timeout is None else timeout
    per_channel = max(1, config.WEB_SEARCH_RETRY_ATTEMPTS)
    backoff = max(0.0, config.WEB_SEARCH_RETRY_BACKOFF)
    attempts = 0
    last_kind = ""

    for backend, plan_query in _search_plan(query):
        for attempt in range(1, per_channel + 1):
            attempts += 1
            # 默认参数绑定当前这一轮的通道与检索式：lambda 在守护线程里执行，
            # 直接引用循环变量会读到「之后的值」（B023），变成用错参数发请求。
            ok, text, kind = _run_with_status(
                lambda b=backend, q=plan_query: _search_once(b, q, max_results),
                timeout,
                "联网检索",
            )
            if ok and (text or "").strip():
                logger.info("联网检索完成 | %d 字 | %s", len(text or ""), summarize(query))
                metrics.incr("tool.call.ok")
                breaker.record(_SEARCH_TOOL_NAME, True)
                return ToolOutcome(text=f"{fetch_time_prefix()}\n{text}".strip(), attempts=attempts)
            last_kind = kind or "empty"
            if last_kind not in RETRYABLE_ERRORS:
                break  # 不可重试的错：同通道再试没有意义，直接进入下一策略
            if attempt < per_channel and backoff:
                time.sleep(backoff)

    logger.warning("联网检索最终降级 | %s | 分类=%s | 尝试=%d", summarize(query), last_kind, attempts)
    metrics.incr("tool.call.fail")
    metrics.incr(f"tool.fail.{last_kind or 'empty'}")
    breaker.record(_SEARCH_TOOL_NAME, False)
    return ToolOutcome(error_kind=last_kind or "empty", attempts=attempts)


def web_search(query: str, max_results: int = 5, timeout: int | None = None) -> str:
    """联网检索（兼容旧签名）：失败返回空串并降级为纯 LLM 回答。

    自带超时保护：底层客户端在无网络时可能长时间阻塞，这里用守护线程限制最长等待
    `timeout` 秒（默认取 config.WEB_SEARCH_TIMEOUT），超时即降级，避免界面卡在「正在思考」。
    分级细节（重试 / 换参数 / 换通道）见 `web_search_detailed`；这里保持
    「拿不到就返回空串」的旧契约，不影响既有调用点。
    """
    return web_search_detailed(query, max_results=max_results, timeout=timeout).text


def _document_time(chunk: Any) -> str:
    """推断知识库片段的文档时间（没有则明说「未知」）。

    优先用入库时记录的元信息；缺失时按来源路径的文件修改时间推断 ——
    宁可标注「未知」，也不要让模型把过时资料当最新事实。
    """
    meta = getattr(chunk, "meta", None) or {}
    for key in ("updated_at", "mtime", "modified", "indexed_at"):
        value = meta.get(key)
        if value:
            return str(value)[:16]
    source = str(getattr(chunk, "source", "") or "")
    if source:
        for candidate in (Path(source), config.KNOWLEDGE_DIR / source, config.SAMPLES_DIR / source):
            try:
                if candidate.is_file():
                    return datetime.fromtimestamp(candidate.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                continue
    return "未知"


def _knowledge_search_tool(kb_name: str, retrieval_cfg=None):
    """构造知识库检索工具（延迟导入，避免与 rag 包循环依赖）。

    `retrieval_cfg` 可显式指定检索配置；不传则用流水线默认值。
    评测场景会传入与显式路径**完全相同**的配置，保证 A/B 对比的是「链路」而非「检索参数」。
    """
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def knowledge_search(query: str) -> str:
        """在指定知识库中检索与问题相关的资料片段（结果含来源、页码与文档时间）。"""
        from .rag.registry import get_registry

        pipeline = get_registry().get(kb_name)
        if pipeline is None:
            return f"知识库 {kb_name} 不存在或尚未建库。"
        # 与联网检索同等待遇：向量化 + 检索同样可能因网络挂起，超时即降级
        chunks = _run_with_timeout(
            lambda: pipeline.retrieve(query, retrieval_cfg),
            config.WEB_SEARCH_TIMEOUT,
            None,
            "知识库检索",
        )
        if not chunks:
            return "未检索到相关内容。"
        return "\n\n".join(
            f"[{i + 1}] 来源：{c.source} 第{c.page or '-'}页（文档时间：{_document_time(c)}）\n{c.text}"
            for i, c in enumerate(chunks)
        )

    return knowledge_search


def _safe_web_search_tool():
    """构造联网检索工具（供 bind_tools 使用）。

    内部复用 `web_search_detailed()`，因此与显式检索路径共享同一套
    「超时 + 失败分级 + 降级」语义。失败时不抛错，而是回灌一句带分级的中文事实说明，
    避免异常细节进入上下文推高 token，同时让模型知道「这次没查到」而不是「没有这事」。
    """
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def search_web(query: str) -> str:
        """联网检索公开资料，适用于需要最新信息或站外知识的问题。"""
        outcome = web_search_detailed(query)
        if outcome.ok:
            return outcome.text
        return _degraded_text(outcome.error_kind or "unknown")

    return search_web


def get_agent_tools(
    kb_name: str | None = None, *, include_web: bool = True, retrieval_cfg=None
) -> list:
    """返回可供 bind_tools 使用的工具列表（联网检索 + 指定知识库检索）。

    `include_web=False` 供「只允许查内部资料」的场景使用（如知识库问答），
    避免联网结果混入答案后污染引用来源。
    `retrieval_cfg` 用于把知识库工具钉在指定检索配置上（评测 A/B 对齐用）。
    """
    tools: list = []
    if include_web and config.ENABLE_WEB_SEARCH:
        try:
            from langchain_community.tools import DuckDuckGoSearchResults  # noqa: F401

            tools.append(_safe_web_search_tool())
        except Exception as exc:  # noqa: BLE001
            logger.debug("搜索工具不可用：%s", exc)

    if kb_name:
        try:
            tools.append(_knowledge_search_tool(kb_name, retrieval_cfg))
        except Exception as exc:  # noqa: BLE001
            logger.debug("知识库工具不可用：%s", exc)
    return tools
