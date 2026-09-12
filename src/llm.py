"""大模型接入层：统一 OpenAI 兼容协议 + 重试 + 超时。

业务代码只依赖本模块，不感知具体厂商（DeepSeek / 通义 / OpenAI / 硅基流动）。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from . import config
from .cache import get_cost_tracker
from .logging_setup import get_logger, summarize
from .telemetry import span

logger = get_logger(__name__)

# 按 (模型名, 温度, 是否流式) 分别缓存客户端：多模型路由场景下
# 首次构建后复用，零重复构建开销；与原单例行为一致。
_MODEL_CACHE: dict[tuple, ChatOpenAI] = {}


def get_chat_model(temperature: float | None = None, streaming: bool = False,
                   model: str | None = None,
                   max_tokens: int | None = None) -> ChatOpenAI:
    """获取（并缓存）聊天模型客户端；支持按模型名 / 输出上限分别构建与缓存。

    ``max_tokens`` 为输出 token 硬上限（控制生成长度 9.2.A）：仅当非 None 才透传给
    SDK，避免把 None 传给 ChatOpenAI；关闭开关或上游传 None 时行为与本项前一致。
    """
    name = model or config.MODEL_NAME
    key = (name, temperature, streaming, max_tokens)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    if not config.llm_ready():
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY。请复制 .env.example 为 .env 并填入密钥后重启。"
        )
    kwargs: dict = {
        "model": name,
        "api_key": config.OPENAI_API_KEY,
        "base_url": config.OPENAI_BASE_URL,
        "temperature": temperature if temperature is not None else config.TEMPERATURE,
        "timeout": config.REQUEST_TIMEOUT,
        "max_retries": 0,  # 重试统一由 tenacity 控制，避免双重重试
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    client = ChatOpenAI(**kwargs)
    _MODEL_CACHE[key] = client
    logger.info(
        "LLM 初始化完成 | model=%s base=%s max_tokens=%s",
        name, config.OPENAI_BASE_URL, max_tokens,
    )
    return client


def reset_chat_model() -> None:
    """清空缓存的客户端（用于配置热更新或测试）。"""
    _MODEL_CACHE.clear()


def _is_retryable(exc: BaseException) -> bool:
    """仅重试瞬态错误：网络连通异常、服务端 5xx。

    非瞬态错误（4xx：含 429 额度耗尽 / 401 鉴权 / 400 参数）以及我们转换出的
    友好 RuntimeError 一律不重试，避免白白消耗已耗尽的额度并让用户久等。
    """
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", 0) >= 500
    if isinstance(exc, RuntimeError):
        return False
    return True


def _cache_tokens_from_usage(usage) -> tuple[int, int]:
    """从 OpenAI 兼容 usage 解析 (缓存命中 tokens, 新建缓存 tokens)，无则返回 (0, 0)。

    兼容两种形态：DashScope/OpenAI 原生对象（usage.prompt_tokens_details.cached_tokens）
    与 langchain 归一化后的 dict（usage_metadata / response_metadata.token_usage）。
    """
    if not usage:
        return 0, 0
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cached = details.get("cached_tokens") or details.get("cache_read_input_tokens") or 0
        created = (
            details.get("cache_creation_input_tokens")
            or usage.get("cache_creation_input_tokens")
            or 0
        )
        return int(cached or 0), int(created or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    created = getattr(usage, "cache_creation_input_tokens", 0) or getattr(
        details, "cache_creation_input_tokens", 0
    ) or 0
    return int(cached or 0), int(created or 0)


def _record_cache_usage(usage, sp) -> None:
    """解析并上报缓存命中/创建 token 到 CostTracker 与当前 span。"""
    cached, created = _cache_tokens_from_usage(usage)
    if cached or created:
        get_cost_tracker().record_cache(cached, created)
        if sp is not None:
            sp.set_attribute("llm.cache_read_tokens", cached)
            sp.set_attribute("llm.cache_creation_tokens", created)


@retry(
    retry=retry_if_exception(predicate=_is_retryable),
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def llm_invoke(messages: Sequence[BaseMessage], temperature: float | None = None,
               model: str | None = None,
               max_tokens: int | None = None) -> str:
    """调用模型并返回文本；瞬态错误自动指数退避，额度耗尽等给出友好提示。"""
    with span(
        "llm.invoke",
        {
            "llm.model": model or config.MODEL_NAME,
            "llm.base_url": config.OPENAI_BASE_URL,
            "llm.temperature": temperature if temperature is not None else config.TEMPERATURE,
            "llm.max_tokens": max_tokens if max_tokens is not None else -1,
            "llm.input_chars": len(str(messages[-1].content)) if messages else 0,
        },
    ) as sp:
        model = get_chat_model(temperature=temperature, model=model, max_tokens=max_tokens)
        try:
            resp = model.invoke(list(messages))
        except RateLimitError as exc:
            # 免费额度耗尽 / 触发限流：不重试，给出可操作的提示
            raise RuntimeError(
                "DashScope 额度已耗尽或触发限流，请更换 API Key 或稍后重试"
            ) from exc
        except APIStatusError as exc:
            if getattr(exc, "status_code", 0) < 500:
                raise RuntimeError(
                    f"模型调用失败（HTTP {exc.status_code}）：{getattr(exc, 'message', str(exc))}"
                ) from exc
            raise
        except Exception as exc:  # noqa: BLE001
            sp.set_attribute("error", True)
            sp.set_attribute("error.type", type(exc).__name__)
            raise
        _record_cache_usage(
            (resp.response_metadata or {}).get("token_usage")
            or getattr(resp, "usage_metadata", None),
            sp,
        )
        content = getattr(resp, "content", "")
        if isinstance(content, list):  # 部分兼容接口返回 list
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        sp.set_attribute("llm.completion_chars", len(content or ""))
        logger.debug("LLM 返回 %d 字 | %s", len(content or ""), summarize(str(messages[-1].content)))
        return content or ""


def llm_stream(messages: Sequence[BaseMessage], temperature: float | None = None,
               model: str | None = None,
               max_tokens: int | None = None) -> Iterator[str]:
    """流式调用模型，逐段产出文本。"""
    with span(
        "llm.stream",
        {
            "llm.model": model or config.MODEL_NAME,
            "llm.base_url": config.OPENAI_BASE_URL,
            "llm.temperature": temperature if temperature is not None else config.TEMPERATURE,
            "llm.max_tokens": max_tokens if max_tokens is not None else -1,
        },
    ) as sp:
        model = get_chat_model(temperature=temperature, model=model, max_tokens=max_tokens)
        try:
            stream_model = model.bind(stream_options={"include_usage": True})
        except Exception:  # noqa: BLE001
            stream_model = model
        last_usage = None
        for chunk in stream_model.stream(list(messages)):
            um = getattr(chunk, "usage_metadata", None)
            if um:
                last_usage = um
            rm = getattr(chunk, "response_metadata", None) or {}
            if rm.get("token_usage"):
                last_usage = rm["token_usage"]
            text = getattr(chunk, "content", "")
            if text:
                yield text
        _record_cache_usage(last_usage, sp)


def check_llm_health() -> tuple[bool, str]:
    """探测模型连通性，返回 (是否可用, 说明)。"""
    with span("llm.health", {"llm.model": config.MODEL_NAME}):
        try:
            from langchain_core.messages import HumanMessage

            text = llm_invoke([HumanMessage(content="回复两个字：正常")])
            return True, (text or "").strip()[:50] or "连通正常"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"


def build_messages(system: str, history: Sequence[BaseMessage]) -> list[BaseMessage]:
    """在系统人设前拼接历史消息。"""
    from langchain_core.messages import SystemMessage

    return [SystemMessage(content=system), *history]
