"""数据清洗：去除页眉页脚、空白、重复行、乱码与目录行。

对应 JD 职责 3「数据清洗」与要求 6「基本的数据清洗能力」。
策略保持可解释（每条规则都有明确动机），便于面试时讲清为什么这么洗。
"""

from __future__ import annotations

import re

from ..logging_setup import get_logger
from .types import RawDocument

logger = get_logger(__name__)

# 常见页眉页脚模式
_HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*第?\s*\d+\s*页?(\s*/\s*共?\s*\d+\s*页?)?\s*$"),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),
]

# 目录行：以一串点结尾并跟页码
_TOC_PATTERN = re.compile(r"^.{2,60}?[.．·]{4,}\s*\d{1,4}\s*$")

# 控制字符与不可见字符
_CTRL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 疑似乱码：大量连续不可识别符号
_MOJIBAKE_PATTERN = re.compile(r"[�]{2,}")


def clean_text(text: str) -> str:
    """清洗单段文本。"""
    if not text:
        return ""

    text = _CTRL_PATTERN.sub("", text)
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = _MOJIBAKE_PATTERN.sub("", text)

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            continue
        if any(p.match(stripped) for p in _HEADER_FOOTER_PATTERNS):
            continue
        if _TOC_PATTERN.match(stripped):
            continue
        # 纯符号 / 纯数字短行（常见于分隔线）
        if len(stripped) <= 3 and not re.search(r"[\u4e00-\u9fffA-Za-z]", stripped):
            continue
        lines.append(line)

    # 合并连续重复行（PDF 解析常出现重复文本层）
    deduped: list[str] = []
    for line in lines:
        if deduped and deduped[-1] == line:
            continue
        deduped.append(line)

    result = "\n".join(deduped)
    # 压缩 3 个以上连续空行为 2 个
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def clean_documents(docs: list[RawDocument]) -> list[RawDocument]:
    """批量清洗，丢弃清洗后为空的文档。"""
    cleaned: list[RawDocument] = []
    removed = 0
    for doc in docs:
        before = len(doc.text or "")
        text = clean_text(doc.text)
        if not text:
            removed += 1
            continue
        doc.text = text
        doc.meta["raw_len"] = before
        doc.meta["clean_len"] = len(text)
        cleaned.append(doc)
    if removed:
        logger.info("清洗阶段丢弃空文档 %d 个", removed)
    return cleaned
