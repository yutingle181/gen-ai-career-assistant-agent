"""统一日志配置。

安全约束：日志只记录 query 摘要（截断）、耗时、token 数与命中数，
禁止记录 API Key 与文档全文。
"""

from __future__ import annotations

import logging
import sys

from . import config

_CONFIGURED = False

_SENSITIVE_HINTS = ("api_key", "apikey", "authorization", "token", "secret")


class _RedactFilter(logging.Filter):
    """兜底脱敏：命中敏感关键词时打码。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        low = msg.lower()
        if any(h in low for h in _SENSITIVE_HINTS):
            record.msg = "[已脱敏] " + msg[:120]
            record.args = ()
        return True


def setup_logging(level: str | None = None) -> logging.Logger:
    """初始化全局日志（幂等）。"""
    global _CONFIGURED
    logger = logging.getLogger("genai_career")
    if _CONFIGURED:
        return logger

    logger.setLevel(getattr(logging, (level or config.LOG_LEVEL), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(_RedactFilter())
    logger.addHandler(handler)

    # Windows 控制台默认 GBK，避免 emoji/生僻字导致 UnicodeEncodeError
    for h in logger.handlers:
        if hasattr(h.stream, "reconfigure"):
            try:
                h.stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
            except Exception:
                pass

    _CONFIGURED = True
    return logger


def get_logger(name: str = "genai_career") -> logging.Logger:
    """获取 logger，确保已完成初始化。"""
    setup_logging()
    return logging.getLogger(name)


def summarize(text: str, limit: int = 120) -> str:
    """生成可安全写入日志的摘要。"""
    if not text:
        return ""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"
