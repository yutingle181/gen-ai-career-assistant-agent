"""链路追踪：未启用 OTel 时必须完全 no-op 且不报错。

这是「零侵入埋点」的核心保证——业务代码无条件调用 span()，
没装 otel 包或没配端点时都不能抛异常、不能影响主流程。
"""

from __future__ import annotations

from src import telemetry


def test_span_with_attributes_is_noop_safe():
    with telemetry.span("unit.test", {"k": "v"}) as sp:
        assert sp is not None
        sp.set_attribute("k2", 1)


def test_span_without_attributes():
    with telemetry.span("unit.test.bare") as sp:
        assert sp is not None


def test_setup_telemetry_is_idempotent():
    telemetry.setup_telemetry()
    telemetry.setup_telemetry()


def test_noop_fallback_when_otel_unavailable(monkeypatch):
    """OTel 不可导入时，span/setup 必须退化为 no-op 且不报错（零侵入兜底保证）。"""
    import importlib
    import sys

    import src.telemetry as tel

    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    importlib.reload(tel)
    try:
        tel.setup_telemetry()
        with tel.span("fallback.test", {"k": "v"}) as sp:
            assert sp is not None
            sp.set_attribute("k2", 1)
    finally:
        monkeypatch.undo()
        importlib.reload(tel)

