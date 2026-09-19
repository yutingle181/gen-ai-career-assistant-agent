"""数据时效（P0-3 收尾）：不只是「印一个时间」，而是把「过没过期」算好告诉模型。

为什么必须由工具算：模型没有时钟，也**不擅长日期减法**。
只给一句「文档时间：2023-05-01」，它常常照样把三年前的数字当成当前事实；
给出「距今 1236 天，可能已过期」这种**结论式标注**，遵守率才上得去。
本模块是 `prompts/personas.FRESHNESS_RULE` 的配套物：规则负责「必须声明」，
这里负责「把事实算好、写成模型无法忽略的样子」。

另附「时间跨度提示」：一次检索里同时命中同一来源的多个版本（时间跨度很大）时，
工具**不替模型做语义合并**，而是把跨度摆出来，让它自己分辨版本差异。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config

#: 可解析的时间格式（入库元信息 / 文件 mtime / 人工填写的常见写法）。
_TIME_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
)

#: 无法判断时效时的统一措辞（prompt 规则依赖这句话的语义）。
UNKNOWN_LABEL = "未知（无法判断时效，勿默认其最新）"


def parse_time(value: Any) -> datetime | None:
    """尽最大努力把各种写法解析成时间；解析不了返回 None（宁缺勿猜）。

    支持 datetime、Unix 时间戳（秒 / 毫秒）与常见字符串格式。
    刻意**不做模糊解析**（如「去年」「近期」）：猜错比说「未知」更危险。
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        stamp = float(value)
        if stamp > 1e11:  # 毫秒时间戳
            stamp /= 1000.0
        try:
            return datetime.fromtimestamp(stamp)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit() and len(text) >= 10:  # 纯数字串＝时间戳（19 位雪花 id 不会落在这里）
        return parse_time(int(text))
    for fmt in _TIME_FORMATS:
        for candidate in (text, text[:16], text[:10]):
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


def age_days(value: Any, *, now: datetime | None = None) -> int | None:
    """距今天数；无法解析返回 None（未来时间返回负数，交由调用方表述）。"""
    stamp = parse_time(value)
    if stamp is None:
        return None
    return ((now or datetime.now()) - stamp).days


def annotate(value: Any, *, now: datetime | None = None) -> tuple[str, bool]:
    """渲染成「结论式标注」，返回 `(标签, 是否不可信/可能过期)`。

    三种形态：
    - 过期：`2023-05-01（距今 1236 天，可能已过期）` → True
    - 新鲜：`2026-08-20（距今 30 天）` → False
    - 解析不了：`未知（无法判断时效，勿默认其最新）` → False（未知 ≠ 过期，
      但 prompt 规则要求对它同样作声明）

    第二个返回值统一表示「**不能默认这是最新资料**」，因此未来时间（时间异常）
    也计为 True —— 它同样不应该被当作当下事实使用。
    """
    stamp = parse_time(value)
    if stamp is None:
        return UNKNOWN_LABEL, False
    days = age_days(stamp, now=now)
    assert days is not None  # parse_time 成功则 days 必可算出
    label_date = stamp.strftime("%Y-%m-%d")
    if days < 0:
        return f"{label_date}（时间晚于当前 {abs(days)} 天，疑似时间异常）", True
    if days > max(0, config.FRESHNESS_STALE_DAYS):
        return f"{label_date}（距今 {days} 天，可能已过期）", True
    return f"{label_date}（距今 {days} 天）", False


def span_note(values: Iterable[Any]) -> str:
    """多版本提示：资料时间跨度较大时，提醒模型分辨版本而不是合并结论。

    语义合并（「哪个版本才是对的」）属于模型与人，工具不做；
    但工具必须把「这次拿到的资料时间差很大」这个**事实**摆出来，
    否则模型很容易把两个年代的结论揉成一句「目前的情况是」。
    """
    stamps = [t for t in (parse_time(v) for v in values) if t is not None]
    if len(stamps) < 2:
        return ""
    oldest, newest = min(stamps), max(stamps)
    span = (newest - oldest).days
    if span < max(0, config.FRESHNESS_SPAN_DAYS):
        return ""
    return (
        f"⚠ 本次检索到的资料时间跨度 {span} 天（{oldest:%Y-%m-%d} ~ {newest:%Y-%m-%d}），"
        "可能来自不同版本：若结论互相矛盾，请以较新者为准并说明这一差异。"
    )


def chunk_time(chunk: Any) -> str:
    """取知识库片段的**原始**文档时间（取不到就返回「未知」）。

    放在本模块而不是工具层：工具（注入给模型）与流水线（数据侧健康度统计）都要用它，
    两边共用一份取法，才不会出现「工具说这份资料过期、统计说它新鲜」这种自相矛盾。
    取法优先级：入库元信息 → 来源文件 mtime（连文件都找不到就老实说「未知」）。
    """
    meta = getattr(chunk, "meta", None) or {}
    for key in ("updated_at", "mtime", "modified", "indexed_at"):
        value = meta.get(key)
        if value:
            return str(value)[:16]
    source = str(getattr(chunk, "source", "") or "")
    if source:
        for candidate in (Path(source), config.KNOWLEDGE_DIR / source, config.SAMPLES_DIR / source):
            try:
                if candidate.is_file():
                    return datetime.fromtimestamp(candidate.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                continue
    return "未知"


def audit(values: Sequence[Any], *, now: datetime | None = None) -> dict[str, int]:
    """统计一批时间值的时效分布（数据侧健康度：策略问题不该只靠模型兜）。

    返回 `{"total", "stale", "unknown", "fresh"}`；`total` 指提供了时间值的条数。
    """
    stale = unknown = fresh = 0
    for value in values:
        if value in (None, ""):
            continue
        _label, suspect = annotate(value, now=now)
        if parse_time(value) is None:
            unknown += 1
        elif suspect:
            stale += 1
        else:
            fresh += 1
    return {"total": stale + unknown + fresh, "stale": stale, "unknown": unknown, "fresh": fresh}
