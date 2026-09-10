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
from .logging_setup import get_logger, summarize
from .telemetry import span

logger = get_logger(__name__)

_CHAT_MODEL: ChatOpenAI | None = None


def get_chat_model(temperature: float | None = None, streaming: bool = False) -> ChatOpenAI:
    """获取（并缓存）聊天模型客户端。"""
    global _CHAT_MODEL
    if _CHAT_MODEL is None:
        if not config.llm_ready():
            raise RuntimeError(
                "未检测到 OPENAI_API_KEY。请复制 .env.example 为 .env 并填入密钥后重启。"
            )
        _CHAT_MODEL = ChatOpenAI(
            model=config.MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            temperature=config.TEMPERATURE,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=0,  # 重试统一由 tenacity 控制，避免双重重试
        )
        logger.info(
            "LLM 初始化完成 | model=%s base=%s", config.MODEL_NAME, config.OPENAI_BASE_URL
        )

    if temperature is not None and temperature != config.TEMPERATURE:
        return _CHAT_MODEL.bind(temperature=temperature)
    return _CHAT_MODEL


def reset_chat_model() -> None:
    """清空缓存的客户端（用于配置热更新或测试）。"""
    global _CHAT_MODEL
    _CHAT_MODEL = None


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


@retry(
    retry=retry_if_exception(predicate=_is_retryable),
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def llm_invoke(messages: Sequence[BaseMessage], temperature: float | None = None) -> str:
    """调用模型并返回文本；瞬态错误自动指数退避，额度耗尽等给出友好提示。"""
    with span(
        "llm.invoke",
        {
            "llm.model": config.MODEL_NAME,
            "llm.base_url": config.OPENAI_BASE_URL,
            "llm.temperature": temperature if temperature is not None else config.TEMPERATURE,
            "llm.input_chars": len(str(messages[-1].content)) if messages else 0,
        },
    ) as sp:
        model = get_chat_model(temperature=temperature)
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
        content = getattr(resp, "content", "")
        if isinstance(content, list):  # 部分兼容接口返回 list
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        sp.set_attribute("llm.completion_chars", len(content or ""))
        logger.debug("LLM 返回 %d 字 | %s", len(content or ""), summarize(str(messages[-1].content)))
        return content or ""


def llm_stream(messages: Sequence[BaseMessage], temperature: float | None = None) -> Iterator[str]:
    """流式调用模型，逐段产出文本。"""
    with span(
        "llm.stream",
        {
            "llm.model": config.MODEL_NAME,
            "llm.base_url": config.OPENAI_BASE_URL,
            "llm.temperature": temperature if temperature is not None else config.TEMPERATURE,
        },
    ):
        model = get_chat_model(temperature=temperature)
        for chunk in model.stream(list(messages)):
            text = getattr(chunk, "content", "")
            if text:
                yield text


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
