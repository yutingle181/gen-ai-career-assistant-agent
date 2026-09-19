"""Skills 骨架：注册 / 版本 / 权限。

三件事都必须可断言，否则"骨架"就只是文档：
- 注册表覆盖全部场景，且**不改变既有行为**（平价检查）；
- 接口主版本不兼容时拒绝注册并回退，而不是带病运行；
- 权限收紧后工具**真的不挂载**、显式检索**真的不发**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src import config, metrics, tools
from src.agents import AGENT_CLASSES, create_agent
from src.skills import REGISTRY, Skill, register_defaults
from src.skills.registry import SKILL_API_VERSION
from src.state import MODE_KNOWLEDGE, MODE_QA, MODE_TUTORIAL

FAKE_MCP = Path(__file__).with_name("fake_mcp_server.py")

_CONFIG_KEYS = ("SKILL_PERMISSIONS", "SKILLS_DISABLED", "SKILL_ALLOW_API_MISMATCH", "ENABLE_MCP")


@pytest.fixture(autouse=True)
def _registry_isolation():
    """注册表是进程级单例：用例改了它（或配置）必须在结束后还原，否则污染其他文件。"""
    saved = {key: getattr(config, key) for key in _CONFIG_KEYS}
    metrics.reset()
    yield
    for key, value in saved.items():
        setattr(config, key, value)
    REGISTRY.reset()
    register_defaults()
    tools.get_breaker().reset()
    metrics.reset()


def _reload_registry(**overrides) -> None:
    """改配置后重建默认注册表（模拟一次冷启动）。"""
    for key, value in overrides.items():
        setattr(config, key, value)
    REGISTRY.reset()
    metrics.reset()
    register_defaults()


# ---------------------------------------------------------------- 注册
def test_default_registry_covers_all_agent_modes():
    for mode, cls in AGENT_CLASSES.items():
        skill = REGISTRY.for_mode(mode)
        assert skill is not None, mode
        assert skill.agent_cls is cls
        assert skill.version.count(".") == 2, "版本必须是 x.y.z"
        assert skill.api_version == SKILL_API_VERSION


def test_registry_is_single_source_of_truth_for_agent_classes():
    resolved = {mode: REGISTRY.for_mode(mode).agent_cls for mode in REGISTRY.modes()}
    assert resolved == AGENT_CLASSES


def test_create_agent_parity_with_table():
    """平价检查：骨架上线后每个模式创建的 Agent 类型与原来一致。"""
    for mode, cls in AGENT_CLASSES.items():
        agent = create_agent(mode)
        assert isinstance(agent, cls), mode


def test_unknown_mode_falls_back_to_qa_agent():
    agent = create_agent("no-such-mode")
    assert agent.mode == MODE_QA
    assert metrics.counter("skill.fallback") == 1


def test_duplicate_name_rejected():
    skill = Skill(name="qa.general", mode="qa2", agent_cls=AGENT_CLASSES[MODE_QA])
    result = REGISTRY.register(skill)
    assert result.ok is False
    assert "已注册" in result.reason
    assert metrics.counter("skill.register_rejected") == 1


def test_api_version_mismatch_rejected_by_default():
    """接口主版本不兼容必须拒绝：宁可少一个场景，也不要带不确定的契约跑。"""
    skill = Skill(
        name="legacy.scene",
        mode="legacy",
        agent_cls=AGENT_CLASSES[MODE_QA],
        version="1.0.0",
        api_version="2.0",
    )

    result = REGISTRY.register(skill)

    assert result.ok is False
    assert "接口版本不兼容" in result.reason
    assert REGISTRY.get("legacy.scene") is None


def test_api_version_mismatch_can_be_allowed_explicitly(monkeypatch):
    monkeypatch.setattr(config, "SKILL_ALLOW_API_MISMATCH", True)
    skill = Skill(
        name="legacy.scene",
        mode="legacy",
        agent_cls=AGENT_CLASSES[MODE_QA],
        version="1.0.0",
        api_version="2.0",
    )
    assert REGISTRY.register(skill).ok is True


def test_missing_contract_members_rejected():
    class _Broken:
        mode = "broken"

    result = REGISTRY.register(Skill(name="broken.scene", mode="broken", agent_cls=_Broken))

    assert result.ok is False
    assert "契约成员" in result.reason


def test_bad_version_format_rejected():
    result = REGISTRY.register(
        Skill(name="bad.version", mode="bad", version="v1", agent_cls=AGENT_CLASSES[MODE_QA])
    )
    assert result.ok is False


def test_disabled_skill_falls_back_to_default_scene():
    _reload_registry(SKILLS_DISABLED="knowledge.qa")

    assert REGISTRY.for_mode(MODE_KNOWLEDGE) is None
    assert create_agent(MODE_KNOWLEDGE).mode == MODE_QA  # 回退，而不是报错
    assert metrics.counter("skill.disabled") == 1


# ---------------------------------------------------------------- 权限
def test_effective_permissions_is_intersection():
    """知识库场景声明 kb：全局就算给了 net/mcp，它也只拿到 kb（最小权限）。"""
    _reload_registry(SKILL_PERMISSIONS="net,kb,mcp")
    assert REGISTRY.effective_permissions(MODE_KNOWLEDGE) == frozenset({"kb"})

    _reload_registry(SKILL_PERMISSIONS="net")
    assert REGISTRY.effective_permissions(MODE_KNOWLEDGE) == frozenset()


def test_unregistered_mode_gets_no_permission():
    assert REGISTRY.effective_permissions("no-such-mode") == frozenset()


def test_kb_tool_not_mounted_when_permission_revoked():
    _reload_registry(SKILL_PERMISSIONS="net")  # 收回 kb
    agent = create_agent(MODE_KNOWLEDGE, kb_name="data1")

    names = [t.name for t in agent.build_tools()]

    assert "knowledge_search" not in names
    assert metrics.counter("skill.permission_denied") >= 1


def test_kb_tool_mounted_with_default_permissions():
    agent = create_agent(MODE_KNOWLEDGE, kb_name="data1")
    names = [t.name for t in agent.build_tools()]
    assert "knowledge_search" in names


def test_web_tool_not_mounted_without_net_permission():
    _reload_registry(SKILL_PERMISSIONS="kb")
    from src.skills import effective_permissions

    names = [
        t.name
        for t in tools.get_agent_tools(
            "data1", include_web=True, permissions=effective_permissions(MODE_TUTORIAL)
        )
    ]

    assert "search_web" not in names  # tutorial 要 net，但全局已收回
    assert metrics.counter("skill.permission_denied") >= 1


def test_explicit_search_respects_net_permission(monkeypatch):
    """权限必须同时覆盖「显式前置检索」：否则收紧权限只关掉一半。"""
    calls: list = []
    monkeypatch.setattr(tools, "web_search", lambda q, **k: calls.append(q) or "检索结果")
    _reload_registry(SKILL_PERMISSIONS="kb")  # 收回 net
    agent = create_agent(MODE_TUTORIAL)

    assert agent.prepare("问题") == ""
    assert calls == []
    assert metrics.counter("skill.permission_denied") >= 1

    _reload_registry(SKILL_PERMISSIONS="net")
    agent2 = create_agent(MODE_TUTORIAL)
    assert "联网检索结果" in agent2.prepare("问题")
    assert calls == ["问题"]


def test_mcp_tools_require_mcp_permission(monkeypatch):
    """MCP 单独一类权限：外部工具风险面更大，可以单独收口。"""
    monkeypatch.setattr(config, "ENABLE_MCP", True)
    monkeypatch.setattr(config, "MCP_CONFIG_PATH", "")
    monkeypatch.setattr(config, "MCP_MAX_TOOLS", 12)
    spec_args = {
        "fake": {
            "command": sys.executable,
            "args": [str(FAKE_MCP)],
        }
    }
    from src.mcp.client import parse_specs

    monkeypatch.setattr(
        "src.mcp.tools.load_specs", lambda *a, **k: parse_specs({"mcpServers": spec_args})
    )
    _reload_registry(SKILL_PERMISSIONS="net,kb")  # 未授予 mcp

    names = [t.name for t in tools.get_agent_tools(permissions=frozenset({"net", "kb"}))]

    assert not any(n.startswith("mcp_") for n in names)
    assert metrics.counter("skill.permission_denied") >= 1

    from src.mcp import tools as mcp_tools

    mcp_tools.close_clients()


# ---------------------------------------------------------------- 快照
def test_snapshot_exposes_registry_and_grants():
    from src.skills import snapshot

    snap = snapshot()

    assert snap["api_version"] == SKILL_API_VERSION
    assert "kb" in snap["granted_permissions"]
    assert len(snap["skills"]) == len(AGENT_CLASSES)
    assert all({"name", "version", "permissions", "agent"} <= set(s) for s in snap["skills"])


def test_default_grants_keep_capability_surface_unchanged():
    """默认授权面必须与改造前一致（否则就是"顺手"收了口径）。"""
    from src.skills import granted_permissions

    assert granted_permissions() == frozenset({"net", "kb", "mcp"})
    assert REGISTRY.effective_permissions("job_search") == frozenset({"net", "kb"})
