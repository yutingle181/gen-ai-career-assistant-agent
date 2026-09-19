"""工具熔断（P2-7）：连续失败即短路、冷却后半开、成功即复位。

回归背景：重试只能对付「单次抖动」。上游整体挂掉时，没有熔断的后果是
每一次用户提问都要把重试次数走满——延迟成倍增加，结果依然为空。
"""

from __future__ import annotations

import time

import pytest

from src import config, metrics, tools


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """默认打开熔断、阈值 2、不等待退避，且每个用例前清空状态。"""
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)
    monkeypatch.setattr(config, "ENABLE_TOOL_CIRCUIT_BREAKER", True)
    monkeypatch.setattr(config, "TOOL_BREAKER_THRESHOLD", 2)
    monkeypatch.setattr(config, "TOOL_BREAKER_COOLDOWN", 60)
    monkeypatch.setattr(config, "WEB_SEARCH_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(config, "WEB_SEARCH_RETRY_BACKOFF", 0.0)
    metrics.reset()
    tools.get_breaker().reset()
    yield
    tools.get_breaker().reset()
    metrics.reset()


def _always_fail(monkeypatch) -> None:
    def _boom(backend: str, query: str, max_results: int) -> str:
        raise ConnectionError("network down")

    monkeypatch.setattr(tools, "_search_once", _boom)


# ------------------------------------------------------------------ 打开与短路
def test_breaker_opens_after_threshold_and_short_circuits(monkeypatch):
    _always_fail(monkeypatch)

    assert tools.web_search_detailed("问题一").error_kind == "network"
    assert tools.web_search_detailed("问题二").error_kind == "network"

    third = tools.web_search_detailed("问题三")
    assert third.error_kind == "circuit_open"
    assert third.attempts == 0  # 直接短路：一次上游请求都没发
    assert tools.get_breaker().snapshot()["open"] == ["search_web"]
    assert metrics.counter("tool.call.blocked") == 1


def test_breaker_can_be_disabled(monkeypatch):
    """开关关闭时行为与改动前一致：每次都真实调用（零回归路径）。"""
    monkeypatch.setattr(config, "ENABLE_TOOL_CIRCUIT_BREAKER", False)
    _always_fail(monkeypatch)

    for _ in range(5):
        assert tools.web_search_detailed("问题").error_kind == "network"
    assert metrics.counter("tool.call.blocked") == 0


# ------------------------------------------------------------------ 半开与复位
def test_cooldown_expiry_half_opens(monkeypatch):
    """冷却结束后必须放一次真实调用过去（否则上游恢复了也永远短路）。"""
    breaker = tools.get_breaker()
    _always_fail(monkeypatch)
    tools.web_search_detailed("a")
    tools.web_search_detailed("b")
    assert breaker.is_open("search_web") is True

    breaker._open_until["search_web"] = time.monotonic() - 0.01  # 模拟冷却已结束
    assert breaker.is_open("search_web") is False

    out = tools.web_search_detailed("c")
    assert out.error_kind == "network"  # 放过去了，且这次真的打了上游


def test_success_resets_failure_count(monkeypatch):
    breaker = tools.get_breaker()
    _always_fail(monkeypatch)
    tools.web_search_detailed("a")
    assert breaker.snapshot()["failures"]["search_web"] == 1

    monkeypatch.setattr(tools, "_search_once", lambda backend, query, n: "检索结果")
    assert tools.web_search_detailed("b").ok is True
    assert breaker.snapshot()["failures"] == {}


# ------------------------------------------------------------------ 用户可见文案
def test_circuit_open_message_is_actionable():
    """熔断提示必须能被写进答案：说明「没拿到外部资料」且要求声明未联网核实。"""
    text = tools.degraded_text("circuit_open")
    assert tools.degraded_kind(text) == "circuit_open"
    assert "熔断中" in text and "未联网核实" in text
