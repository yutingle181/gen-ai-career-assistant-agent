"""工具层：联网搜索（可降级）+ 供 Function Calling 使用的工具定义。

对应 JD 要求 3「Function Calling」：本模块同时提供
- `web_search()`：显式检索（默认路径，稳定、低延迟）；
- `get_agent_tools()`：bind_tools 工具调用版本（由 `ENABLE_TOOL_CALLING` 开关启用）。

两者并存本身就是一个可讲的技术决策。关键约定：两条路径**共享同一套**「守护线程
超时 + 失败即降级」实现——此前工具列表里直接放入裸 `DuckDuckGoSearchResults`，
那条路径没有超时保护，与显式检索路径行为不一致，弱网下会把整轮对话拖住。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

from . import config
from .logging_setup import get_logger, summarize

logger = get_logger(__name__)

_T = TypeVar("_T")


def _run_with_timeout(fn: Callable[[], _T], timeout: int, default: _T, what: str) -> _T:
    """在守护线程中执行 `fn`：超时或抛异常一律返回 `default`，绝不阻塞主流程。

    工具调用链路里「慢」比「错」更伤体验——底层客户端在弱网下可能长时间挂起，
    因此所有外部工具统一走这一实现，保证「同一件事只有一种行为」。
    """
    box: dict = {}

    def _run() -> None:
        try:
            box["r"] = fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s失败，降级 | %s | %s", what, type(exc).__name__, exc)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.warning("%s超时（%ss），降级", what, timeout)
        return default
    return box.get("r", default)


def web_search(query: str, max_results: int = 5, timeout: int | None = None) -> str:
    """DuckDuckGo 检索，失败时返回空串并降级为纯 LLM 回答。

    自带超时保护：底层客户端在无网络时可能长时间阻塞，这里用守护线程
    限制最长等待 `timeout` 秒（默认取 config.WEB_SEARCH_TIMEOUT），超时即
    降级为空串，避免界面一直卡在「正在思考」。
    """
    if not config.ENABLE_WEB_SEARCH:
        logger.info("联网搜索已关闭 | %s", summarize(query))
        return ""
    timeout = config.WEB_SEARCH_TIMEOUT if timeout is None else timeout

    def _search() -> str:
        from langchain_community.tools import DuckDuckGoSearchResults

        tool = DuckDuckGoSearchResults(num_results=max_results)
        return tool.run(query) or ""

    result = _run_with_timeout(_search, timeout, "", "联网检索")
    if result:
        logger.info("联网检索完成 | %d 字 | %s", len(result), summarize(query))
    return result


def _knowledge_search_tool(kb_name: str, retrieval_cfg=None):
    """构造知识库检索工具（延迟导入，避免与 rag 包循环依赖）。

    `retrieval_cfg` 可显式指定检索配置；不传则用流水线默认值。
    评测场景会传入与显式路径**完全相同**的配置，保证 A/B 对比的是「链路」而非「检索参数」。
    """
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def knowledge_search(query: str) -> str:
        """在指定知识库中检索与问题相关的资料片段。"""
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
            f"[{i + 1}] 来源：{c.source} 第{c.page or '-'}页\n{c.text}" for i, c in enumerate(chunks)
        )

    return knowledge_search


def _safe_web_search_tool():
    """构造联网检索工具（供 bind_tools 使用）。

    内部复用 `web_search()`，因此与显式检索路径共享同一套超时与降级语义。
    工具无结果时不抛错，而是回灌一句中文提示，避免异常细节进入上下文推高 token。
    """
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def search_web(query: str) -> str:
        """联网检索公开资料，适用于需要最新信息或站外知识的问题。"""
        result = web_search(query)
        if not result:
            return "联网检索无结果或暂不可用，请基于已有信息作答。"
        return result

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
