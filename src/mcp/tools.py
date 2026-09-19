"""把 MCP 服务端的工具包装成本项目的 LangChain 工具。

两个关键设计（都是踩过才知道值得写下来的）：

1. **名字必须改写**：MCP 工具名允许含 `/`（如 `get/weather`），而模型侧的 function name
   只接受 `[A-Za-z0-9_-]`，直接透传会被上游拒掉。统一改成 `mcp_<server>_<tool>`：
   前缀让来源在事件流与答案里一眼可辨，也避免不同 server 的同名工具互相覆盖；
   调用时再映射回**原始**工具名。
2. **失败沿用同一套契约**：服务端 `isError`、连接不可用、超时、协议错误分别给不同 kind，
   统一回灌带 `【工具降级:xxx】` 标记的中文说明。这样工具闭环的事件流、轨迹指标
   （Failure Onset）与熔断统计**自动覆盖** MCP 工具，而不是另起一套。
"""

from __future__ import annotations

import re
import threading
from typing import Any

from pydantic import Field, create_model

from .. import config, metrics
from ..logging_setup import get_logger
from ..tools import degraded_text, get_breaker
from .client import MCPClient, MCPError, MCPServerSpec, load_specs

logger = get_logger(__name__)

_NAME_SAFE = re.compile(r"[^A-Za-z0-9_-]")
MAX_NAME_LEN = 64

_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

_CLIENTS: dict[str, MCPClient] = {}
_LOCK = threading.Lock()
#: 最近一次 build 的失败原因（/health 用；只给结论，不含命令与 env）
_LAST_ERRORS: dict[str, str] = {}


def safe_tool_name(server: str, tool: str) -> str:
    """把「服务端 / 工具名」改写成模型侧合法的 function name（≤64 字符）。"""
    raw = f"mcp_{server}_{tool}"
    return _NAME_SAFE.sub("_", raw)[:MAX_NAME_LEN]


def _args_schema(tool_name: str, input_schema: Any):
    """由 MCP 的 `inputSchema` 生成 pydantic 参数模型。

    为什么必须生成 schema：模型要靠它生成 function 定义。用 `**kwargs` 蒙混过去，
    模型就不知道有哪些参数、该怎么填 —— 那等于工具挂上了却不可用。
    解析不了的字段按「可选 + Any」处理，宁可宽松也不要让一个怪 schema 让工具整个消失。
    """
    schema = input_schema if isinstance(input_schema, dict) else {}
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = {str(item) for item in (schema.get("required") or [])}
    fields: dict[str, Any] = {}
    for prop_name, raw in props.items():
        info = raw if isinstance(raw, dict) else {}
        py_type = _JSON_TO_PY.get(str(info.get("type") or "string"), Any)
        desc = str(info.get("description") or "")[:200]
        if str(prop_name) in required:
            fields[str(prop_name)] = (py_type, Field(description=desc))
        else:
            fields[str(prop_name)] = (py_type | None, Field(default=None, description=desc))
    if not fields:
        return None  # 无参工具：给 None 让 LangChain 生成空参数模型
    return create_model(f"{tool_name}_args", **fields)


def get_client(spec: MCPServerSpec) -> MCPClient:
    """按名字复用连接：stdio 进程有启动成本，不该每次工具调用都重启一次。"""
    with _LOCK:
        client = _CLIENTS.get(spec.name)
        if client is not None and client.alive:
            return client
        if client is not None:
            logger.warning("MCP 连接已失效，重新建立 | %s", spec.label())
            client.close()
        fresh = MCPClient(spec)
        fresh.start()
        _CLIENTS[spec.name] = fresh
        return fresh


def close_clients() -> None:
    """关闭全部 MCP 连接（应用退出 / 测试清理用）。"""
    with _LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
    for client in clients:
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("关闭 MCP 连接失败（忽略）：%s", exc)


def snapshot() -> dict[str, Any]:
    """MCP 运行时快照（/health 用）。

    **刻意不含 command 与 env**：env 里常放 token / 数据库口令，
    健康检查接口是能对外暴露的，不该顺带把凭据泄出去。
    """
    with _LOCK:
        servers = [
            {
                "name": name,
                "alive": client.alive,
                "protocol": client.negotiated_version,
                "server_info": client.server_info.get("name") or "",
            }
            for name, client in _CLIENTS.items()
        ]
    return {
        "enabled": config.ENABLE_MCP,
        "config_path": config.MCP_CONFIG_PATH,
        "max_tools": config.MCP_MAX_TOOLS,
        "protocol_version": config.MCP_PROTOCOL_VERSION,
        "servers": servers,
        "last_errors": dict(_LAST_ERRORS),
    }


def _wrap_tool(client: MCPClient, server: str, tool_def: dict[str, Any]):
    """把一个 MCP 工具包装成 LangChain 工具。"""
    from langchain_core.tools import StructuredTool

    original = str(tool_def.get("name") or "")
    name = safe_tool_name(server, original)
    description = str(tool_def.get("description") or f"{server} 提供的外部工具")
    breaker_key = f"mcp:{server}"

    def _run(**kwargs: Any) -> str:
        breaker = get_breaker()
        if breaker.is_open(breaker_key):
            metrics.incr("mcp.tool.blocked")
            logger.warning("MCP 服务端处于熔断冷却期，直接短路 | %s", server)
            return degraded_text("circuit_open", channel="mcp")
        # 把 None 去掉：MCP 服务端通常不接受显式 null 参数
        arguments = {k: v for k, v in kwargs.items() if v is not None}
        result = client.call_tool(original, arguments)
        breaker.record(breaker_key, result.ok)
        if result.ok:
            return result.text or "（工具返回空结果）"
        return degraded_text(result.error_kind or "mcp_error", channel="mcp")

    _run.__name__ = name
    return StructuredTool(
        name=name,
        description=f"[MCP/{server}] {description}",
        args_schema=_args_schema(name, tool_def.get("inputSchema")),
        func=_run,
    )


def build_mcp_tools(specs: dict[str, MCPServerSpec] | None = None) -> list:
    """连接（或复用）所有 MCP 服务端，返回可 `bind_tools` 的工具列表。

    单个服务端出问题**不影响**其他服务端与主流程 —— 与项目里「工具的坏不该拖垮对话」
    同一口径；失败原因记进 `_LAST_ERRORS`，在 `/health` 里可见。
    """
    resolved = specs if specs is not None else load_specs()
    if not resolved:
        return []

    tools: list = []
    for server_name, spec in resolved.items():
        try:
            client = get_client(spec)
        except MCPError as exc:
            _LAST_ERRORS[server_name] = f"连接失败：{exc}"
            logger.warning("MCP 服务端不可用，已跳过 | %s | %s", spec.label(), exc)
            continue
        try:
            tool_defs = client.list_tools()
        except MCPError as exc:
            _LAST_ERRORS[server_name] = f"tools/list 失败：{exc}"
            logger.warning("MCP tools/list 失败，已跳过 | %s | %s", spec.label(), exc)
            continue
        _LAST_ERRORS.pop(server_name, None)
        for tool_def in tool_defs:
            if len(tools) >= max(1, config.MCP_MAX_TOOLS):
                logger.warning(
                    "MCP 工具数已达上限 %d，其余未挂载（函数定义会进每轮 prompt）",
                    config.MCP_MAX_TOOLS,
                )
                logger.info("MCP 工具就绪 | %d 个 | 服务端 %d 个", len(tools), len(resolved))
                return tools
            tools.append(_wrap_tool(client, server_name, tool_def))

    logger.info("MCP 工具就绪 | %d 个 | 服务端 %d 个", len(tools), len(resolved))
    return tools
