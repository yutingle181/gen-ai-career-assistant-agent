"""MCP 客户端（stdio 传输，JSON-RPC 2.0）。

为什么手写而不是引第三方 SDK：项目既有约定是「不新增依赖、不引入新框架」，
而 stdio 这条传输的协议面其实很小：换行分隔的 JSON-RPC + `initialize` 握手 +
`tools/list` + `tools/call`。协议细节按 MCP 规范核对过：

- stdio 下每条消息必须是**单行** JSON-RPC（换行分隔、不得含内嵌换行）；
- `initialize` 客户端发 `protocolVersion / capabilities / clientInfo`，服务端回
  `protocolVersion / capabilities / serverInfo`，随后客户端发 `notifications/initialized`；
- **工具自身出错要放在 result 里 + `isError=true`**，不能用协议级 error ——
  后者模型看不到，也就无法自我纠正；协议级 error 只留给「找不到工具 / 服务端不支持」这类情况；
- 关闭：先关 stdin，等进程退出，必要时再终止。

能力边界（如实说）：
- 只实现 **tools** 能力（resources / prompts / sampling 未实现）；
- 只实现 **stdio** 传输（HTTP/SSE 传输未实现）；
- 单连接内**串行**调用（一把锁），没有请求并发；
- 服务端若把日志写进 **stdout** 会污染协议：这里跳过解析不了的行并记 debug 日志，
  但规范要求服务端把日志写到 stderr。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config, metrics
from ..logging_setup import get_logger

logger = get_logger(__name__)

CLIENT_NAME = "genai-career-assistant"

#: 协议级兜底：单次请求最多等多久（秒），由 config.MCP_TIMEOUT 覆盖
DEFAULT_TIMEOUT = 20.0

#: tools/list 分页保护：最多翻多少页，避免服务端光标异常导致死循环
MAX_LIST_PAGES = 10


class MCPError(RuntimeError):
    """协议级错误（服务端返回 error、握手失败、进程起不来等）。"""


class MCPTimeout(MCPError):
    """等不到响应（会由工具层翻译成「超时降级」，不会冒泡给用户）。"""


@dataclass
class MCPServerSpec:
    """一个 MCP 服务端的启动描述（对应 mcpServers 配置里的一项）。"""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None

    def label(self) -> str:
        """日志里用的短标识（命令可能含密钥，不进日志）。"""
        return f"{self.name}({Path(self.command).name})"


@dataclass
class MCPToolResult:
    """一次工具调用的结果（与项目其他工具同契约：**永不抛异常**）。"""

    name: str
    text: str = ""
    is_error: bool = False
    error_kind: str = ""
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return not self.is_error


def parse_specs(data: Any) -> dict[str, MCPServerSpec]:
    """解析 `{"mcpServers": {...}}` 配置（兼容 Claude 等客户端的既有格式）。

    容错优先：单项不合法就跳过该项并记警告，不要因为一个坏配置让整个 MCP 失效。
    """
    servers = (data or {}).get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        logger.warning("MCP 配置缺少 mcpServers 段，已跳过")
        return {}
    specs: dict[str, MCPServerSpec] = {}
    for name, raw in servers.items():
        if not isinstance(raw, dict):
            logger.warning("MCP 服务端 %s 配置不是对象，已跳过", name)
            continue
        if raw.get("disabled"):
            logger.info("MCP 服务端 %s 已禁用，跳过", name)
            continue
        command = str(raw.get("command") or "").strip()
        if not command:
            logger.warning("MCP 服务端 %s 缺少 command，已跳过", name)
            continue
        args = [str(a) for a in (raw.get("args") or [])]
        env = {str(k): str(v) for k, v in (raw.get("env") or {}).items()}
        specs[str(name)] = MCPServerSpec(
            name=str(name),
            command=command,
            args=args,
            env=env,
            cwd=str(raw["cwd"]) if raw.get("cwd") else None,
        )
    return specs


def load_specs(path: str | None = None) -> dict[str, MCPServerSpec]:
    """从配置文件读取 MCP 服务端清单；未配置 / 读取失败一律返回空（不影响主流程）。"""
    target = (path or config.MCP_CONFIG_PATH or "").strip()
    if not target:
        return {}
    config_path = Path(target)
    if not config_path.is_file():
        logger.warning("MCP 配置文件不存在：%s", config_path)
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP 配置文件解析失败（%s）：%s", config_path, exc)
        return {}
    return parse_specs(data)


class MCPClient:
    """一个 MCP 服务端的连接（进程 + 握手 + 请求/响应）。

    线程模型：标准输出由一个后台线程读取、投递到队列；请求方持锁等自己那条响应。
    这样超时是可控的（`queue.get(timeout=...)`），而不是卡在 `readline()` 上 ——
    「等不到响应」必须是**可降级**的情况，不能让工具把对话一起拖死。
    """

    def __init__(
        self,
        spec: MCPServerSpec,
        *,
        timeout: float | None = None,
        protocol_version: str | None = None,
    ) -> None:
        self.spec = spec
        self.timeout = float(timeout if timeout is not None else config.MCP_TIMEOUT)
        self.requested_version = protocol_version or config.MCP_PROTOCOL_VERSION
        self.negotiated_version = ""
        self.server_info: dict[str, Any] = {}
        self.instructions = ""
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._closed = False

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        """启动服务端进程并完成 initialize 握手。"""
        if self._proc is not None:
            return
        env = {**os.environ, **self.spec.env}
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - 命令来自用户自己的配置文件
                _argv(self.spec.command, self.spec.args),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self.spec.cwd or None,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            metrics.incr("mcp.server.start_fail")
            raise MCPError(f"启动失败：{type(exc).__name__}: {exc}") from exc

        self._reader = threading.Thread(target=self._read_stdout, name=f"mcp-{self.spec.name}", daemon=True)
        self._reader.start()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name=f"mcp-{self.spec.name}-err", daemon=True
        )
        self._stderr_thread.start()

        result = self._request(
            "initialize",
            {
                "protocolVersion": self.requested_version,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": CLIENT_NAME, "version": config_version()},
            },
        )
        self.negotiated_version = str(result.get("protocolVersion") or self.requested_version)
        self.server_info = result.get("serverInfo") or {}
        self.instructions = str(result.get("instructions") or "")
        if self.negotiated_version != self.requested_version:
            # 规范要求：服务端可以回一个它支持的版本；客户端不支持就断开。
            # 这里选择「接受并记日志」：tools 能力在近几版协议里是稳定的，
            # 直接断开对用户更糟（而是把事实写进日志，便于排查）。
            logger.info(
                "MCP 协议版本协商：请求 %s → 服务端 %s | %s",
                self.requested_version,
                self.negotiated_version,
                self.spec.label(),
            )
        self._notify("notifications/initialized")
        metrics.incr("mcp.server.start")
        logger.info(
            "MCP 服务端已连接 | %s | protocol=%s | server=%s",
            self.spec.label(),
            self.negotiated_version,
            self.server_info.get("name") or "unknown",
        )

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and not self._closed

    def close(self) -> None:
        """关闭连接：先关 stdin（规范的关停信号），再兜底终止进程。"""
        if self._proc is None:
            return
        self._closed = True
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                self._proc.kill()
        finally:
            self._proc = None

    def __enter__(self) -> MCPClient:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------ 能力
    def list_tools(self) -> list[dict[str, Any]]:
        """列出服务端工具（按规范处理 `nextCursor` 分页）。"""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MAX_LIST_PAGES):
            params: dict[str, Any] = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params)
            for item in result.get("tools") or []:
                if isinstance(item, dict) and item.get("name"):
                    tools.append(item)
            cursor = result.get("nextCursor") or None
            if not cursor:
                break
        else:
            logger.warning("MCP tools/list 分页超过 %d 页，已截断 | %s", MAX_LIST_PAGES, self.spec.label())
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        """调用一个工具；**任何失败都返回结果对象**，不抛异常（工具层的统一契约）。"""
        started = time.monotonic()
        metrics.incr("mcp.tool.call")
        try:
            result = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        except MCPTimeout as exc:
            return self._failure(name, "timeout", str(exc), started)
        except MCPError as exc:
            # 协议级错误（服务端 error / 进程已退出）——注意这与 isError 是两码事
            kind = "mcp_unavailable" if not self.alive else "mcp_protocol"
            return self._failure(name, kind, str(exc), started)
        except Exception as exc:  # noqa: BLE001
            return self._failure(name, "mcp_protocol", f"{type(exc).__name__}: {exc}", started)

        text = _content_text(result)
        is_error = bool(result.get("isError"))
        elapsed = int((time.monotonic() - started) * 1000)
        if is_error:
            # 工具自己报错：按规范它就在 result 里，照原样回灌即可（模型看得到就能自我纠正）
            metrics.incr("mcp.tool.fail")
            metrics.incr("mcp.tool.fail.mcp_error")
            logger.warning("MCP 工具返回错误 | %s/%s | %s", self.spec.name, name, text[:120])
            return MCPToolResult(
                name=name, text=text, is_error=True, error_kind="mcp_error", elapsed_ms=elapsed
            )
        metrics.incr("mcp.tool.ok")
        logger.info("MCP 工具完成 | %s/%s | %dms | %d 字", self.spec.name, name, elapsed, len(text))
        return MCPToolResult(name=name, text=text, elapsed_ms=elapsed)

    def _failure(self, name: str, kind: str, detail: str, started: float) -> MCPToolResult:
        metrics.incr("mcp.tool.fail")
        metrics.incr(f"mcp.tool.fail.{kind}")
        logger.warning("MCP 工具失败 | %s/%s | 分类=%s | %s", self.spec.name, name, kind, detail)
        return MCPToolResult(
            name=name,
            text=detail,
            is_error=True,
            error_kind=kind,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    # ------------------------------------------------------------ 传输
    def _send(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise MCPError("连接未建立")
        if self._proc.poll() is not None:
            raise MCPError(f"服务端进程已退出（code={self._proc.returncode}）")
        line = json.dumps(payload, ensure_ascii=False)
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            raise MCPError(f"写入失败：{type(exc).__name__}: {exc}") from exc

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """发请求并等响应（带超时）；返回 result 段，error 段一律转成 MCPError。"""
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
            deadline = time.monotonic() + max(1.0, self.timeout)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPTimeout(f"{method} 超时（{self.timeout:.0f}s）")
                try:
                    message = self._queue.get(timeout=remaining)
                except queue.Empty:
                    raise MCPTimeout(f"{method} 超时（{self.timeout:.0f}s）") from None
                if message.get("id") != request_id:
                    # 通知或其他请求的响应（当前实现串行，正常不会出现，出现就跳过）
                    logger.debug("忽略无关 MCP 消息：%s", str(message)[:120])
                    continue
                if "error" in message:
                    err = message["error"] or {}
                    raise MCPError(str(err.get("message") or err))
                result = message.get("result")
                return result if isinstance(result, dict) else {}

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _read_stdout(self) -> None:
        """后台读取：单行 JSON-RPC 投递到队列；非 JSON 行只记 debug（服务端日志应走 stderr）。"""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            text = (line or "").strip()
            if not text:
                continue
            try:
                self._queue.put(json.loads(text))
            except json.JSONDecodeError:
                logger.debug("MCP stdout 非 JSON 行（已忽略）：%s", text[:200])

    def _drain_stderr(self) -> None:
        """排空 stderr：不排空会把子进程写日志阻塞住（经典死锁）。"""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            text = (line or "").strip()
            if text:
                logger.debug("MCP[%s] %s", self.spec.name, text[:300])


def _argv(command: str, args: list[str]) -> list[str]:
    """构造真正要 spawn 的 argv。

    Windows 上 `.cmd` / `.bat`（例如 `npx.cmd`）**不能**被 CreateProcess 直接执行，
    必须经 `cmd.exe /c`；这一条是踩过的：官方 filesystem server 的启动命令就是 `npx.cmd`，
    不包装会直接 `FileNotFoundError` / `not a valid Win32 application`。
    """
    if os.name == "nt" and command.lower().endswith((".cmd", ".bat")):
        shell = os.environ.get("COMSPEC") or "cmd.exe"
        return [shell, "/c", command, *args]
    return [command, *args]


def _content_text(result: dict[str, Any]) -> str:
    """把 `content` 块拼成可回灌的文本。

    非文本块（图片 / 资源）**只留占位摘要**：把 base64 塞进上下文既贵又没用，
    而且会挤掉真正有用的信息。
    """
    blocks = result.get("content")
    if isinstance(blocks, str):  # 宽松兼容：有的服务端直接给字符串
        return blocks
    if not isinstance(blocks, list):
        structured = result.get("structuredContent")
        return json.dumps(structured, ensure_ascii=False)[:4000] if structured else ""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        kind = str(block.get("type") or "")
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind == "image":
            parts.append("[图片内容已省略]")
        elif kind in {"resource", "resource_link"}:
            uri = block.get("uri") or (block.get("resource") or {}).get("uri")
            parts.append(f"[资源：{uri}]")
        else:
            parts.append(f"[{kind or 'unknown'} 内容已省略]")
    return "\n".join(p for p in parts if p).strip()


def config_version() -> str:
    """客户端版本（握手用）。延迟导入避免与包初始化顺序纠缠。"""
    from .. import __version__

    return __version__
