"""日志模块：摘要脱敏与初始化幂等。

日志是「禁止记录 API Key 与文档全文」这条安全约束的最后一道兜底，需有回归测试。
"""

from __future__ import annotations

import logging

from src.logging_setup import _RedactFilter, get_logger, setup_logging, summarize


def test_summarize_flattens_whitespace():
    assert summarize("  a\n  b  ") == "a b"


def test_summarize_truncates_with_ellipsis():
    out = summarize("x" * 200)
    assert len(out) == 121
    assert out.endswith("…")


def test_summarize_empty():
    assert summarize("") == ""


def test_setup_logging_is_idempotent():
    assert setup_logging() is setup_logging()


def test_redact_filter_masks_api_key():
    record = logging.LogRecord("n", logging.INFO, "p", 1, "api_key=sk-1234567890abcdef", None, None)
    assert _RedactFilter().filter(record) is True
    assert "已脱敏" in str(record.msg)


def test_redact_filter_keeps_normal_message():
    msg = "检索完成 | 命中 3 条"
    record = logging.LogRecord("n", logging.INFO, "p", 1, msg, None, None)
    assert _RedactFilter().filter(record) is True
    assert record.getMessage() == msg


def test_get_logger_returns_named_logger():
    assert get_logger("tests.unit").name == "tests.unit"
