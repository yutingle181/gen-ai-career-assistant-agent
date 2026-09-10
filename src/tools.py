"""工具层：联网搜索（可降级）+ 供 Function Calling 使用的工具定义。

对应 JD 要求 3「Function Calling」：本模块同时提供
- `web_search()`：显式检索（默认路径，稳定、低延迟）；
- `get_agent_tools()`：bind_tools / AgentExecutor 版本（可配置开启）。
两者并存本身就是一个可讲的技术决策。
"""

from __future__ import annotations

import threading

from . import config
from .logging_setup import get_logger, summarize

logger = get_logger(__name__)


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

    box: dict = {}

    def _run() -> None:
        try:
            from langchain_community.tools import DuckDuckGoSearchResults

            tool = DuckDuckGoSearchResults(num_results=max_results)
            box["r"] = tool.run(query) or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("联网检索失败，降级为纯模型回答 | %s | %s", type(exc).__name__, exc)
            box["r"] = ""

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.warning("联网检索超时（%ss），降级为纯模型回答 | %s", timeout, summarize(query))
        return ""
    result = box.get("r", "")
    logger.info("联网检索完成 | %d 字 | %s", len(result), summarize(query))
    return result


def _knowledge_search_tool(kb_name: str):
    """构造知识库检索工具（延迟导入，避免与 rag 包循环依赖）。"""
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def knowledge_search(query: str) -> str:
        """在指定知识库中检索与问题相关的资料片段。"""
        from .rag.registry import get_registry

        pipeline = get_registry().get(kb_name)
        if pipeline is None:
            return f"知识库 {kb_name} 不存在或尚未建库。"
        chunks = pipeline.retrieve(query)
        if not chunks:
            return "未检索到相关内容。"
        return "\n\n".join(
            f"[{i + 1}] 来源：{c.source} 第{c.page or '-'}页\n{c.text}" for i, c in enumerate(chunks)
        )

    return knowledge_search


def get_agent_tools(kb_name: str | None = None) -> list:
    """返回可供 bind_tools 使用的工具列表。"""
    tools: list = []
    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        if config.ENABLE_WEB_SEARCH:
            tools.append(DuckDuckGoSearchResults(num_results=5))
    except Exception as exc:  # noqa: BLE001
        logger.debug("搜索工具不可用：%s", exc)

    if kb_name:
        try:
            tools.append(_knowledge_search_tool(kb_name))
        except Exception as exc:  # noqa: BLE001
            logger.debug("知识库工具不可用：%s", exc)
    return tools
