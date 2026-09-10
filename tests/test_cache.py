"""结果缓存与成本统计。

缓存未启用时必须完全退化为空实现（不报错、不落盘）；
成本统计是界面与日志展示的数据来源，需保证累计口径正确。
"""

from __future__ import annotations

import pytest

from src.cache import (
    CostTracker,
    ResultCache,
    Timer,
    count_tokens,
    get_cache,
    get_cost_tracker,
)


def test_make_key_is_deterministic():
    assert ResultCache.make_key("a", "b") == ResultCache.make_key("a", "b")


def test_make_key_depends_on_order():
    assert ResultCache.make_key("a", "b") != ResultCache.make_key("b", "a")


def test_disabled_cache_is_noop():
    cache = ResultCache(enabled=False)
    cache.set("k", {"a": 1})
    assert cache.get("k") is None
    cache.clear()  # 不应抛异常


def test_enabled_cache_roundtrip(tmp_path):
    pytest.importorskip("diskcache")
    cache = ResultCache(directory=str(tmp_path / "cache"), enabled=True)
    if not cache.enabled:
        pytest.skip("diskcache 初始化失败，跳过")
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    cache.clear()
    assert cache.get("k") is None


def test_count_tokens():
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0


def test_cost_tracker_accumulates():
    tracker = CostTracker()
    tracker.record("prompt", "answer", latency_ms=10)
    tracker.record("another prompt", "answer2", latency_ms=20)
    assert tracker.calls == 2
    assert tracker.prompt_tokens > 0
    assert tracker.completion_tokens > 0
    assert tracker.total_latency_ms == 30


def test_cost_tracker_summary_fields():
    tracker = CostTracker()
    tracker.record("prompt", "answer")
    summary = tracker.summary()
    assert summary["调用次数"] == 1
    assert summary["合计 tokens"] == summary["输入 tokens"] + summary["输出 tokens"]


def test_cost_usd_unknown_model_is_zero():
    tracker = CostTracker()
    tracker.record("hello", "world")
    assert tracker.cost_usd("model-not-in-price-table") == 0.0


def test_cost_usd_known_model_is_positive():
    tracker = CostTracker()
    tracker.record("hello", "world")
    assert tracker.cost_usd("deepseek-chat") > 0


def test_cost_tracker_reset():
    tracker = CostTracker()
    tracker.record("prompt", "answer")
    tracker.reset()
    assert tracker.calls == 0 and tracker.prompt_tokens == 0


def test_timer_measures_elapsed():
    with Timer() as t:
        pass
    assert t.ms >= 0


def test_singletons_are_stable():
    assert get_cache() is get_cache()
    assert get_cost_tracker() is get_cost_tracker()
