"""RAG 链路的公共数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDocument:
    """解析后的原始文档单元（通常一页）。"""

    source: str
    text: str
    page: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """切分后的文本片段，是检索与评测的最小单位。"""

    chunk_id: str
    text: str
    source: str
    page: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cite(self) -> str:
        page = f" 第{self.page}页" if self.page else ""
        return f"{self.source}{page}"
