"""DashScope 上下文缓存（会话/Prompt 缓存）接入测试（确定性、无真实网络）。

覆盖：
- user_context 注入：开关 / 模型白名单 / 最小长度 三道闸门的降级与生效；
- usage 解析：dict 与对象两种形态下缓存命中 / 创建 token 的提取；
- CostTracker：缓存 token 累计与 summary 展示。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import config
from src.cache import CostTracker
from src.llm import _cache_tokens_from_usage
from src.session import _user_context_message

LONG = "岗位 JD 与简历内容" * 200  # 远超最小字符阈值


@pytest.fixture
def _qwen(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_PROMPT_CACHE", True)
    monkeypatch.setattr(config, "MODEL_NAME", "qwen-plus")
    monkeypatch.setattr(config, "PROMPT_CACHE_MIN_CHARS", 1024)


def test_disabled_returns_plain_system_message(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_PROMPT_CACHE", False)
    msg = _user_context_message(LONG)
    assert isinstance(msg.content, str)
    assert msg.content == LONG


def test_qwen_long_is_cacheable(_qwen):
    msg = _user_context_message(LONG)
    assert isinstance(msg.content, list)
    block = msg.content[0]
    assert block["type"] == "text"
    assert block["cache_control"] == {"type": "ephemeral"}


def test_non_qwen_is_plain(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_PROMPT_CACHE", True)
    monkeypatch.setattr(config, "MODEL_NAME", "deepseek-chat")
    msg = _user_context_message(LONG)
    assert isinstance(msg.content, str)


def test_short_context_is_plain(_qwen):
    msg = _user_context_message("太短的档案")
    assert isinstance(msg.content, str)


def test_cache_tokens_from_dict_cached():
    cached, created = _cache_tokens_from_usage(
        {"prompt_tokens_details": {"cached_tokens": 2048}}
    )
    assert (cached, created) == (2048, 0)


def test_cache_tokens_from_dict_creation():
    cached, created = _cache_tokens_from_usage(
        {"prompt_tokens_details": {"cache_creation_input_tokens": 100}}
    )
    assert (cached, created) == (0, 100)


def test_cache_tokens_from_dict_langchain_shape():
    cached, created = _cache_tokens_from_usage(
        {"input_tokens_details": {"cache_read_input_tokens": 50}}
    )
    assert (cached, created) == (50, 0)


def test_cache_tokens_from_object():
    usage = SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(cached_tokens=10, cache_creation_input_tokens=5)
    )
    cached, created = _cache_tokens_from_usage(usage)
    assert (cached, created) == (10, 5)


def test_cache_tokens_empty_is_zero():
    assert _cache_tokens_from_usage(None) == (0, 0)
    assert _cache_tokens_from_usage({}) == (0, 0)


def test_cost_tracker_record_cache_accumulates():
    tracker = CostTracker()
    tracker.record_cache(100, 20)
    tracker.record_cache(50, 0)
    assert tracker.cache_read_tokens == 150
    assert tracker.cache_creation_tokens == 20
    summary = tracker.summary()
    assert summary["缓存命中 tokens"] == 150
    assert summary["缓存创建 tokens"] == 20
