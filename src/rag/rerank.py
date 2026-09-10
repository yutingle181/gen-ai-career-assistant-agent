"""重排层（Rerank）。

对应 JD 职责 3「检索及重排」。三种实现：
- `none`：不重排，直接用融合分（对照组）；
- `llm`：用大模型对候选片段打 0-10 分（零额外依赖，默认，适合无 GPU 环境）；
- `api`：云端 rerank 接口（如硅基流动 bge-reranker-v2-m3），效果更好。

三者构成 A-B 实验组，直接产出评测对比数据。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol

from .. import config
from ..logging_setup import get_logger
from .types import Chunk

logger = get_logger(__name__)


class BaseReranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[Chunk], top_n: int) -> list[tuple[Chunk, float]]:
        ...


class NoneReranker:
    """对照组：保持原顺序，分数沿用融合分。"""

    name = "none"

    def rerank(self, query: str, candidates: Sequence[Chunk], top_n: int) -> list[tuple[Chunk, float]]:
        return [(c, float(top_n - i)) for i, c in enumerate(candidates[:top_n])]


class LLMReranker:
    """用大模型对候选片段按相关性打分。"""

    name = "llm"

    PROMPT = (
        "你是一个检索结果重排助手。给定用户问题和若干候选片段，"
        "请判断每个片段对回答该问题的有用程度，按 0-10 的整数打分（10 分最相关）。\n"
        "只输出 JSON，格式为 {{\"scores\": [{{\"index\": 1, \"score\": 9}}, ...]}}，不要输出其他内容。\n\n"
        "用户问题：{query}\n\n候选片段：\n{candidates}\n"
    )

    def rerank(self, query: str, candidates: Sequence[Chunk], top_n: int) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        from langchain_core.messages import HumanMessage

        from ..llm import llm_invoke

        block = "\n\n".join(
            f"[{i + 1}] 来源：{c.source}{' 第' + str(c.page) + '页' if c.page else ''}\n{c.text[:600]}"
            for i, c in enumerate(candidates)
        )
        prompt = self.PROMPT.format(query=query, candidates=block)
        try:
            raw = llm_invoke([HumanMessage(content=prompt)], temperature=0)
            scores = self._parse(raw, len(candidates))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 重排失败，回退为原顺序 | %s", exc)
            return [(c, float(len(candidates) - i)) for i, c in enumerate(candidates[:top_n])]

        scored = [(candidates[i], float(s)) for i, s in scores.items() if 0 <= i < len(candidates)]
        if not scored:
            return [(c, float(len(candidates) - i)) for i, c in enumerate(candidates[:top_n])]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    @staticmethod
    def _parse(raw: str, size: int) -> dict:
        """从模型输出中解析 JSON 打分。"""
        text = (raw or "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            return {}
        out = {}
        for item in data.get("scores", []):
            try:
                idx = int(item["index"]) - 1
                score = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            out[idx] = score
        return out


class APIReranker:
    """云端 rerank 接口（OpenAI 兼容 /rerank）。"""

    name = "api"

    def rerank(self, query: str, candidates: Sequence[Chunk], top_n: int) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        import requests

        url = config.RERANK_BASE_URL.rstrip("/") + "/rerank"
        headers = {
            "Authorization": f"Bearer {config.RERANK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.RERANK_MODEL,
            "query": query,
            "documents": [c.text[:2000] for c in candidates],
            "top_n": min(top_n, len(candidates)),
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("API 重排失败，回退为原顺序 | %s | %s", type(exc).__name__, exc)
            return [(c, float(len(candidates) - i)) for i, c in enumerate(candidates[:top_n])]

        results = data.get("results", data.get("data", []))
        scored: list[tuple[Chunk, float]] = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score", 0))
            if idx is None:
                continue
            idx = int(idx)
            if 0 <= idx < len(candidates):
                scored.append((candidates[idx], float(score)))
        if not scored:
            return [(c, float(len(candidates) - i)) for i, c in enumerate(candidates[:top_n])]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


def get_reranker(provider: str | None = None):
    """工厂：按名称获取重排器。"""
    name = (provider or config.RERANK_PROVIDER or "llm").lower()
    if name in {"none", "off", "false"}:
        return NoneReranker()
    if name == "api":
        return APIReranker()
    return LLMReranker()
