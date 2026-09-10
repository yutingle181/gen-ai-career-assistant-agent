"""OpenTelemetry 链路追踪（可选、零侵入）。

启用方式：设置环境变量 `OTEL_EXPORTER_OTLP_ENDPOINT`（兼容 OTLP 的后端，
如 Jaeger / Tempo / 阿里云 ARMS / 自建 Collector）即自动初始化 TracerProvider 并导出 span。

不启用时的行为：
- 未安装 `opentelemetry-*` 包：本模块提供 no-op 兜底，所有埋点完全不生效、不报错；
- 已安装但未设置端点：使用 OpenTelemetry 默认的 no-op provider，同样零开销。

因此业务代码可无条件调用 `from src.telemetry import span`，无需关心是否启用。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import trace as _trace

    _HAVE_OTEL = True
except Exception:  # noqa: BLE001
    _HAVE_OTEL = False

SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "genai-career-assistant")
_initialized = False

_logger = logging.getLogger(__name__)


def setup_telemetry() -> None:
    """进程启动时调用一次。仅当安装了 OTel 且配置了 OTLP 端点时才真正启用导出。

    任何环节缺失（未装 otel 包 / 未装 OTLP 导出器 / 未设端点）都安全降级为 no-op，
    不会阻塞进程启动。
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    if not _HAVE_OTEL:
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return  # 不配置端点 = no-op

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        _trace.set_tracer_provider(provider)
        _logger.info("OpenTelemetry 链路追踪已启用 | endpoint=%s", endpoint)
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "OpenTelemetry 导出器不可用（请 pip install -r requirements-otel.txt），链路追踪已关闭：%s",
            exc,
        )


if _HAVE_OTEL:
    tracer = _trace.get_tracer(SERVICE_NAME)
else:
    # opentelemetry 未安装：提供 no-op 兜底，保证业务导入不报错
    class _NoopSpan:
        def set_attribute(self, *args, **kwargs) -> None: ...
        def add_event(self, *args, **kwargs) -> None: ...
        def set_status(self, *args, **kwargs) -> None: ...
        def record_exception(self, *args, **kwargs) -> None: ...

    @contextmanager
    def _noop_span(name: str, *args, **kwargs):  # type: ignore
        yield _NoopSpan()

    class _NoopTracer:
        def start_as_current_span(self, name, *args, **kwargs):
            return _noop_span(name)

        def start_span(self, *args, **kwargs):
            return _NoopSpan()

    tracer = _NoopTracer()


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """轻量 span 上下文管理器；未启用时为 no-op。

    用法：
        with span("llm.invoke", {"llm.model": config.MODEL_NAME}):
            ...
    """
    with tracer.start_as_current_span(name) as sp:
        if attributes:
            for k, v in attributes.items():
                try:
                    sp.set_attribute(k, v)
                except Exception:  # noqa: BLE001
                    pass
        yield sp
