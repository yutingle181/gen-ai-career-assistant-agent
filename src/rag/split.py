"""文本切分：递归字符切分 + 标题感知。

对应 JD 职责 3「文本切分」。chunk_size / chunk_overlap 可配置，
评测阶段会对不同 chunk_size 做网格实验。
"""

from __future__ import annotations

import hashlib

from .. import config
from ..logging_setup import get_logger
from .types import Chunk, RawDocument

logger = get_logger(__name__)

# 中英混排友好的切分优先级
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "；", " ", ""]


def _make_id(source: str, page, index: int, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    page_part = f"p{page}" if page else "p0"
    return f"{source}::{page_part}::{index}::{digest}"


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """对单段文本做递归切分。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )
    return [t for t in splitter.split_text(text or "") if t.strip()]


def split_documents(
    docs: list[RawDocument],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_length: int = 20,
) -> list[Chunk]:
    """把文档切分为片段，过短的片段并入下一个以避免碎片化。"""
    chunks: list[Chunk] = []
    idx = 0
    for doc in docs:
        pieces = split_text(doc.text, chunk_size, chunk_overlap)
        buffer = ""
        for piece in pieces:
            buffer = f"{buffer}\n{piece}".strip() if buffer else piece
            if len(buffer) >= min_length:
                chunks.append(
                    Chunk(
                        chunk_id=_make_id(doc.source, doc.page, idx, buffer),
                        text=buffer,
                        source=doc.source,
                        page=doc.page,
                        meta=dict(doc.meta),
                    )
                )
                idx += 1
                buffer = ""
        if buffer:  # 尾部残余
            if chunks and len(buffer) < min_length:
                chunks[-1].text = f"{chunks[-1].text}\n{buffer}".strip()
            else:
                chunks.append(
                    Chunk(
                        chunk_id=_make_id(doc.source, doc.page, idx, buffer),
                        text=buffer,
                        source=doc.source,
                        page=doc.page,
                        meta=dict(doc.meta),
                    )
                )
                idx += 1

    logger.info("切分完成 | 文档 %d -> 片段 %d", len(docs), len(chunks))
    return chunks
