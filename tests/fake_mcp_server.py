"""测试用的最小 MCP 服务端（stdio，JSON-RPC 2.0）。

刻意**按规范实现**，而不是随手返回固定字符串：这样被测客户端面对的是真实协议形态
（换行分隔的单行报文、initialize 握手、tools/list 的 `nextCursor` 分页、
tools/call 的 `isError` 与协议级 error 的区别），而不是我们彼此约定的私有格式。

由 `tests/test_mcp.py` 以子进程方式拉起，不参与 pytest 收集（文件名不匹配 test_*.py）。

工具清单：
- `echo(text)`         正常返回
- `slow(seconds)`      故意慢，用于测超时降级
- `fail(reason)`       返回 result + `isError=true`（规范里「工具自己出错」的表达方式）
- `boom()`             返回**协议级** error（测客户端的 MCPError 分支）
- `get/weather(city)`  名字含 `/`（测 function name 改写 + 原名映射回 tools/call）
"""

from __future__ import annotations

import json
import sys
import time

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "echo",
        "description": "原样返回输入文本",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要回显的文本"}},
            "required": ["text"],
        },
    },
    {
        "name": "slow",
        "description": "睡一会儿再返回（测超时）",
        "inputSchema": {
            "type": "object",
            "properties": {"seconds": {"type": "number", "description": "睡眠秒数"}},
            "required": ["seconds"],
        },
    },
    {
        "name": "fail",
        "description": "总是失败（以 isError 表达，而非协议级错误）",
        "inputSchema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "失败原因"}},
        },
    },
    {
        "name": "boom",
        "description": "触发协议级 error",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get/weather",
        "description": "名字里带斜杠的工具（测名字改写）",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市"}},
            "required": ["city"],
        },
    },
]


def _send(payload: dict) -> None:
    """发一条**单行** JSON-RPC 报文（stdio 传输的规范要求：不得含内嵌换行）。"""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _text(mid, text: str, *, is_error: bool = False) -> None:
    _send(
        {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            },
        }
    )


def _handle(msg: dict) -> None:
    method = msg.get("method")
    mid = msg.get("id")
    params = msg.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    if method == "initialize":
        _send(
            {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "fake-mcp", "version": "0.1.0"},
                    "instructions": "测试用服务端：只在本地进程间通信。",
                },
            }
        )
        return

    if method in {"notifications/initialized", "notifications/cancelled"}:
        return  # 通知没有响应

    if method == "tools/list":
        cursor = params.get("cursor")
        if not cursor:
            # 第一页只给两条并带 nextCursor：让客户端必须真的处理分页
            _send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS[:2], "nextCursor": "page-2"}})
        else:
            _send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS[2:]}})
        return

    if method == "tools/call":
        name = str(params.get("name") or "")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        if name == "echo":
            _text(mid, f"echo: {args.get('text', '')}")
        elif name == "slow":
            time.sleep(float(args.get("seconds") or 1))
            _text(mid, "慢工具完成")
        elif name == "fail":
            _text(mid, f"执行失败：{args.get('reason') or '原因未知'}", is_error=True)
        elif name == "boom":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "error": {"code": -32603, "message": "内部错误：模拟协议级失败"},
                }
            )
        elif name == "get/weather":
            _text(mid, f"{args.get('city', '未知')}：晴，26℃（来自名字带斜杠的工具）")
        else:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "error": {"code": -32602, "message": f"未知工具：{name}"},
                }
            )
        return

    _send(
        {
            "jsonrpc": "2.0",
            "id": mid,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }
    )


def main() -> int:
    for line in sys.stdin:
        text = (line or "").strip()
        if not text:
            continue
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict):
            _handle(message)
    return 0  # stdin 关闭 = 关停信号


if __name__ == "__main__":
    raise SystemExit(main())
