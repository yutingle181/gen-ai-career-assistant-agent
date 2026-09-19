"""MCP 接入测试：走**真实协议**（stdio + JSON-RPC 2.0），全部离线。

为什么用真服务端脚本而不是打桩：MCP 的价值就在「跨进程协议」本身 ——
握手、分页、`isError` 与协议级 error 的区别、关停语义，打桩全都测不到。
被测的服务端是 `tests/fake_mcp_server.py`（按规范实现的最小实现），
被拉起的是真实子进程。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src import config, metrics, tools
from src.mcp import client as mcp_client
from src.mcp import tools as mcp_tools

SERVER_SCRIPT = Path(__file__).with_name("fake_mcp_server.py")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    metrics.reset()
    tools.get_breaker().reset()
    mcp_tools.close_clients()
    mcp_tools._LAST_ERRORS.clear()
    monkeypatch.setattr(config, "ENABLE_MCP", False)
    monkeypatch.setattr(config, "MCP_CONFIG_PATH", "")
    monkeypatch.setattr(config, "MCP_TIMEOUT", 10.0)
    monkeypatch.setattr(config, "MCP_MAX_TOOLS", 12)
    yield
    mcp_tools.close_clients()
    mcp_tools._LAST_ERRORS.clear()
    tools.get_breaker().reset()
    metrics.reset()


@pytest.fixture
def spec() -> mcp_client.MCPServerSpec:
    return mcp_client.MCPServerSpec(name="fake", command=sys.executable, args=[str(SERVER_SCRIPT)])


def _client(spec, **kwargs) -> mcp_client.MCPClient:
    client = mcp_client.MCPClient(spec, **kwargs)
    client.start()
    return client


# ---------------------------------------------------------------- 配置解析
def test_parse_specs_skips_invalid_entries():
    """坏配置只跳过该项，不让整个 MCP 失效。"""
    specs = mcp_client.parse_specs(
        {
            "mcpServers": {
                "good": {"command": "python", "args": ["-m", "x"], "env": {"A": "1"}},
                "no-command": {"args": ["x"]},
                "disabled": {"command": "python", "disabled": True},
            }
        }
    )

    assert list(specs) == ["good"]
    assert specs["good"].args == ["-m", "x"]
    assert specs["good"].env == {"A": "1"}


def test_parse_specs_requires_mcp_servers_key():
    assert mcp_client.parse_specs({"servers": {}}) == {}


def test_load_specs_from_file(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {"fake": {"command": "python", "args": ["s.py"]}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "MCP_CONFIG_PATH", str(path))

    specs = mcp_client.load_specs()

    assert list(specs) == ["fake"]


def test_load_specs_missing_file_is_honest(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MCP_CONFIG_PATH", str(tmp_path / "nope.json"))
    assert mcp_client.load_specs() == {}


# ---------------------------------------------------------------- 协议
def test_initialize_handshake(spec):
    client = _client(spec)
    try:
        assert client.requested_version == config.MCP_PROTOCOL_VERSION
        assert client.negotiated_version == "2025-06-18"  # 假服务端按规范回的版本
        assert client.server_info.get("name") == "fake-mcp"
        assert "测试用服务端" in client.instructions
        assert client.alive is True
    finally:
        client.close()
    assert client.alive is False


def test_list_tools_handles_pagination(spec):
    """服务端第一页只给 2 条并带 nextCursor，客户端必须真的翻页。"""
    with _client(spec) as client:
        names = [t["name"] for t in client.list_tools()]

    assert names == ["echo", "slow", "fail", "boom", "get/weather"]


def test_call_tool_success(spec):
    with _client(spec) as client:
        result = client.call_tool("echo", {"text": "你好"})

    assert result.ok is True
    assert result.text == "echo: 你好"
    assert metrics.counter("mcp.tool.ok") == 1


def test_iserror_is_not_an_exception(spec):
    """工具自己出错 → `isError=true`（规范要求放在 result 里，模型才看得见）。"""
    with _client(spec) as client:
        result = client.call_tool("fail", {"reason": "上游拒绝"})

    assert result.ok is False
    assert result.error_kind == "mcp_error"
    assert "上游拒绝" in result.text


def test_protocol_error_is_distinguished_from_iserror(spec):
    """协议级 error 与「工具执行出错」是两码事：前者是链路问题，后者是业务结果。"""
    with _client(spec) as client:
        result = client.call_tool("boom", {})

    assert result.ok is False
    assert result.error_kind == "mcp_protocol"
    assert "内部错误" in result.text


def test_timeout_degrades_instead_of_hanging(spec, monkeypatch):
    monkeypatch.setattr(config, "MCP_TIMEOUT", 1.0)
    with _client(spec, timeout=1.0) as client:
        result = client.call_tool("slow", {"seconds": 5})

    assert result.ok is False
    assert result.error_kind == "timeout"
    assert metrics.counter("mcp.tool.fail.timeout") == 1


def test_cmd_script_is_wrapped_for_windows(monkeypatch):
    """Windows 上 `.cmd` / `.bat` 必须经 `cmd.exe /c` 才能 spawn。

    官方 filesystem server 的启动命令就是 `npx.cmd`，不包装会直接起不来。
    """
    from types import SimpleNamespace

    fake_os = SimpleNamespace(name="nt", environ={"COMSPEC": r"C:\Windows\System32\cmd.exe"})
    monkeypatch.setattr(mcp_client, "os", fake_os)

    argv = mcp_client._argv(r"D:\softw\node\npx.cmd", ["-y", "pkg"])
    assert argv == [r"C:\Windows\System32\cmd.exe", "/c", r"D:\softw\node\npx.cmd", "-y", "pkg"]

    fake_os.name = "posix"
    assert mcp_client._argv("./run.sh", []) == ["./run.sh"]


def test_unavailable_server_raises_mcp_error():
    bad = mcp_client.MCPServerSpec(name="bad", command="definitely-not-a-real-binary-xyz")
    with pytest.raises(mcp_client.MCPError):
        mcp_client.MCPClient(bad).start()
    assert metrics.counter("mcp.server.start_fail") == 1


# ---------------------------------------------------------------- 包装成 LangChain 工具
def test_build_tools_sanitizes_names_and_keeps_original(spec):
    """名字含 `/` 的工具必须改名（模型侧不合法），但调用要映射回原名。"""
    tools_list = mcp_tools.build_mcp_tools({"fake": spec})
    names = [t.name for t in tools_list]

    assert "mcp_fake_echo" in names
    assert "mcp_fake_get_weather" in names, names
    assert all("/" not in n for n in names)

    weather = next(t for t in tools_list if t.name == "mcp_fake_get_weather")
    assert "晴" in weather.invoke({"city": "北京"})  # 说明原名映射正确


def test_built_tool_returns_degraded_marker_on_failure(spec):
    """失败要沿用统一降级标记：这样轨迹指标与熔断自动覆盖 MCP 工具。"""
    tools_list = mcp_tools.build_mcp_tools({"fake": spec})
    failing = next(t for t in tools_list if t.name == "mcp_fake_fail")

    out = failing.invoke({"reason": "权限不足"})

    assert tools.degraded_kind(out) == "mcp_error"
    assert "外部 MCP 工具暂不可用" in out
    assert metrics.counter("mcp.tool.fail.mcp_error") == 1


def test_built_tool_has_typed_schema(spec):
    """参数 schema 必须从 inputSchema 生成：否则模型不知道该怎么填。"""
    tools_list = mcp_tools.build_mcp_tools({"fake": spec})
    echo = next(t for t in tools_list if t.name == "mcp_fake_echo")

    assert echo.args_schema is not None
    assert "text" in echo.args_schema.model_fields


def test_max_tools_cap(monkeypatch, spec):
    monkeypatch.setattr(config, "MCP_MAX_TOOLS", 2)
    tools_list = mcp_tools.build_mcp_tools({"fake": spec})
    assert len(tools_list) == 2


def test_breaker_short_circuits_mcp_calls(monkeypatch, spec):
    monkeypatch.setattr(config, "TOOL_BREAKER_THRESHOLD", 1)
    tools_list = mcp_tools.build_mcp_tools({"fake": spec})
    failing = next(t for t in tools_list if t.name == "mcp_fake_fail")

    first = failing.invoke({})
    second = failing.invoke({})

    assert tools.degraded_kind(first) == "mcp_error"
    assert tools.degraded_kind(second) == "circuit_open"
    assert metrics.counter("mcp.tool.blocked") == 1


def test_broken_server_is_skipped_without_breaking_others(spec):
    """一个服务端起不来，不该影响其他服务端，也不该拖垮主流程。"""
    bad = mcp_client.MCPServerSpec(name="bad", command="definitely-not-a-real-binary-xyz")

    tools_list = mcp_tools.build_mcp_tools({"bad": bad, "fake": spec})

    assert any(t.name.startswith("mcp_fake_") for t in tools_list)
    assert "bad" in mcp_tools.snapshot()["last_errors"]


# ---------------------------------------------------------------- 接线与安全
def test_get_agent_tools_includes_mcp_only_when_enabled(monkeypatch, spec, tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fake": {"command": spec.command, "args": spec.args},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "MCP_CONFIG_PATH", str(cfg))

    # 默认关闭：一个 MCP 工具都不挂（也绝不启动任何子进程）
    assert all(not t.name.startswith("mcp_") for t in tools.get_agent_tools())

    monkeypatch.setattr(config, "ENABLE_MCP", True)
    assert any(t.name.startswith("mcp_fake_") for t in tools.get_agent_tools())


def test_health_snapshot_does_not_leak_command_or_env(spec):
    """env 里常放凭据，健康检查接口不该把命令与 env 一起吐出去。"""
    secret = mcp_client.MCPServerSpec(
        name="secret", command=spec.command, args=spec.args, env={"API_TOKEN": "super-secret"}
    )
    mcp_tools.build_mcp_tools({"secret": secret})

    dumped = json.dumps(mcp_tools.snapshot(), ensure_ascii=False)

    assert "super-secret" not in dumped
    assert spec.command not in dumped
    assert "secret" in dumped  # 名字要可见，否则排查时不知道挂了谁
