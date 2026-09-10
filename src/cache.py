"""结果缓存与 token/成本统计。

- 缓存：相同 (query + 知识库版本 + 检索配置) 直接返回，降低重复提问成本；
- 成本：用 tiktoken 估算 token，按内置单价表折算费用，供界面与日志展示。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

from . import config
from .logging_setup import get_logger

logger = get_logger(__name__)

# 单价表（美元 / 百万 token），用于粗略估算；未收录模型按 0 计
_PRICE_TABLE = {
    "deepseek-chat": {"in": 0.27, "out": 1.10},
    "deepseek-reasoner": {"in": 0.55, "out": 2.19},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "qwen-plus": {"in": 0.11, "out": 0.30},
    # qwen-plus 的固定快照版，单价与 qwen-plus 一致
    "qwen-plus-2025-07-28": {"in": 0.11, "out": 0.30},
}


# ---------------------------------------------------------------- 缓存
class ResultCache:
    """基于 diskcache 的结果缓存，未启用时自动退化为空实现。"""

    def __init__(self, directory: str | None = None, enabled: bool | None = None) -> None:
        self.enabled = config.ENABLE_CACHE if enabled is None else enabled
        self._cache = None
        if self.enabled:
            try:
                from diskcache import Cache

                self._cache = Cache(str(directory or config.CACHE_DIR))
            except Exception as exc:  # noqa: BLE001
                logger.warning("缓存初始化失败，已禁用：%s", exc)
                self.enabled = False

    @staticmethod
    def make_key(*parts: Any) -> str:
        raw = json.dumps([str(p) for p in parts], ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any:
        if not self.enabled or self._cache is None:
            return None
        try:
            return self._cache.get(key)
        except Exception:  # noqa: BLE001
            return None

    def set(self, key: str, value: Any, expire: int = 86400) -> None:
        if not self.enabled or self._cache is None:
            return
        try:
            self._cache.set(key, value, expire=expire)
        except Exception as exc:  # noqa: BLE001
            logger.debug("写入缓存失败：%s", exc)

    def clear(self) -> None:
        if self._cache is not None:
            try:
                self._cache.clear()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------- token 统计
class CostTracker:
    """累计 token 消耗与费用，线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.total_latency_ms = 0

    def record(
        self,
        prompt_text: str = "",
        completion_text: str = "",
        model: str | None = None,
        latency_ms: int = 0,
    ) -> None:
        p = count_tokens(prompt_text, model)
        c = count_tokens(completion_text, model)
        with self._lock:
            self.prompt_tokens += p
            self.completion_tokens += c
            self.calls += 1
            self.total_latency_ms += latency_ms

    def cost_usd(self, model: str | None = None) -> float:
        model = model or config.MODEL_NAME
        price = _PRICE_TABLE.get(model)
        if not price:
            return 0.0
        return (
            self.prompt_tokens / 1_000_000 * price["in"]
            + self.completion_tokens / 1_000_000 * price["out"]
        )

    def summary(self) -> dict:
        return {
            "调用次数": self.calls,
            "输入 tokens": self.prompt_tokens,
            "输出 tokens": self.completion_tokens,
            "合计 tokens": self.prompt_tokens + self.completion_tokens,
            "估算费用(USD)": round(self.cost_usd(), 6),
            "累计耗时(ms)": self.total_latency_ms,
        }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0
            self.total_latency_ms = 0


def count_tokens(text: str, model: str | None = None) -> int:
    """估算 token 数；无法编码时按字符数近似。"""
    if not text:
        return 0
    model = model or config.MODEL_NAME
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:  # noqa: BLE001
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        # 中文按约 1.5 字符/token 粗略折算
        return max(1, int(len(text) / 1.5))


class Timer:
    """简易耗时统计上下文管理器。"""

    def __init__(self) -> None:
        self.ms = 0
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = int((time.perf_counter() - self._start) * 1000)


_cache_instance: ResultCache | None = None
_cost_instance: CostTracker | None = None


def get_cache() -> ResultCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResultCache()
    return _cache_instance


def get_cost_tracker() -> CostTracker:
    global _cost_instance
    if _cost_instance is None:
        _cost_instance = CostTracker()
    return _cost_instance
