"""MCP（Model Context Protocol）接入：把外部 server 的工具接成本项目可用的工具。

分层：
- `client.py`：stdio 传输 + JSON-RPC 2.0 会话（initialize / tools/list / tools/call）；
- `tools.py`：把服务端工具包装成 LangChain 工具，并沿用本项目统一的失败分级契约。

设计口径与项目其他部分一致：**默认关闭、显式配置才启用；任何失败都降级成可读说明，
不把异常抛给对话**。
"""

from .client import MCPClient, MCPError, MCPServerSpec, MCPTimeout, load_specs, parse_specs
from .tools import build_mcp_tools, close_clients, snapshot

__all__ = [
    "MCPClient",
    "MCPError",
    "MCPServerSpec",
    "MCPTimeout",
    "build_mcp_tools",
    "close_clients",
    "load_specs",
    "parse_specs",
    "snapshot",
]
