"""槽位记忆（P2-6）：关键约束单独记、每轮重注入、默认关闭。

覆盖四件事：
1. 默认关闭时**完全不影响提示词**（零回归）；
2. 打开后能从用户发言里抽出城市 / 岗位方向 / 时间范围 / 学历；
3. 约束在历史被裁剪后依然注入（这是这个能力存在的唯一理由）；
4. 新值覆盖旧值，且恢复的会话能重新把约束找回来。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src import config
from src.session import SessionManager


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "ENABLE_HUMAN_CONFIRM", False)
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", False)


def _manager(monkeypatch, *, slot_memory: bool, max_history: int | None = None) -> SessionManager:
    monkeypatch.setattr(config, "ENABLE_SLOT_MEMORY", slot_memory)
    if max_history is not None:
        monkeypatch.setattr(config, "MAX_HISTORY", max_history)
    manager = SessionManager("qa")
    manager.agent.respond = lambda history, model=None, max_tokens=None, thread_id=None: "回复"
    return manager


def _prompt_text(manager: SessionManager) -> str:
    return "\n".join(str(getattr(m, "content", "")) for m in manager._trim_history())


def test_slot_memory_off_by_default(monkeypatch):
    """默认关闭：提示词里不出现会话约束，也不做任何抽取。"""
    manager = _manager(monkeypatch, slot_memory=False)
    manager.start("我只要广州的岗位，秋招")

    assert manager.slots == {}
    assert "【会话约束" not in _prompt_text(manager)


def test_slot_memory_extracts_hard_constraints(monkeypatch):
    manager = _manager(monkeypatch, slot_memory=True)
    manager.start("我只要广州的 Java 后端工程师岗位，秋招，本科")

    assert manager.slots["目标城市"] == "广州"
    assert manager.slots["岗位方向"] == "Java 后端工程师"
    assert manager.slots["时间范围"] == "秋招"
    assert manager.slots["学历"] == "本科"

    text = _prompt_text(manager)
    assert "【会话约束（每轮都必须遵守）】" in text
    assert "目标城市：广州" in text


def test_slot_survives_history_trimming(monkeypatch):
    """核心用例：约束在原始发言被裁掉之后仍然注入。"""
    manager = _manager(monkeypatch, slot_memory=True, max_history=1)
    manager.start("我只要广州的岗位")
    for _ in range(3):
        manager.step("继续")

    prompt = manager._trim_history()
    joined = "\n".join(str(getattr(m, "content", "")) for m in prompt)
    assert "目标城市：广州" in joined
    # 证明不是靠「历史里还留着」侥幸保住：原始那句确实已被裁掉
    assert "我只要广州的岗位" not in joined


def test_new_slot_value_overrides_old(monkeypatch):
    """用户改主意要以最新为准，并留下一条变更记录便于排查。"""
    manager = _manager(monkeypatch, slot_memory=True)
    manager.start("我只要广州的岗位")
    manager.step("算了，还是北京吧")

    assert manager.slots["目标城市"] == "北京"
    assert manager.slot_updates == ["目标城市=广州", "目标城市=北京"]
    assert "目标城市：北京" in _prompt_text(manager)


def test_restore_rebuilds_slots(monkeypatch):
    """进程重启后恢复的会话，也要能把约束从历史里重新找回来。"""
    manager = _manager(monkeypatch, slot_memory=True)
    manager.restore([("user", "只看深圳的机会"), ("assistant", "好的")])

    assert manager.slots["目标城市"] == "深圳"
    assert "目标城市：深圳" in _prompt_text(manager)


def test_only_user_messages_are_scanned(monkeypatch):
    """助手自己的话不算约束（否则模型复述一遍就把约束改了）。"""
    manager = _manager(monkeypatch, slot_memory=True)
    manager._update_slots([HumanMessage(content="只想在上海工作")])
    assert manager.slots["目标城市"] == "上海"

    from langchain_core.messages import AIMessage

    manager._update_slots([AIMessage(content="你可以考虑北京的岗位")])
    assert manager.slots["目标城市"] == "上海"
