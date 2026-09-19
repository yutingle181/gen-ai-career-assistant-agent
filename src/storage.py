"""Markdown 产物存储：写入、列表、读取、删除。

沿用原 Notebook 的 `Agent_output/{类型}_{时间戳}.md` 命名习惯。
所有读写强制 utf-8，避免 Windows GBK 报错。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from . import config
from .logging_setup import get_logger

logger = get_logger(__name__)

# 文件名 -> 展示类型
TYPE_LABELS = {
    "Tutorial": ("教程", "📘"),
    "Q&A_Doubt_Session": ("答疑记录", "💬"),
    "Resume": ("简历", "📄"),
    "Interview_questions": ("面试题库", "❓"),
    "Mock_Interview": ("面评记录", "🎤"),
    "Job_search": ("职位清单", "🔎"),
    "Knowledge_QA": ("知识库问答", "📚"),
    "JD_Match": ("JD 匹配诊断", "🎯"),
    "Interview_Review": ("面试复盘", "🧭"),
    "Eval_Report": ("评测报告", "📊"),
}

# 草稿产物后缀：人机协同场景下先落草稿（人工确认后才落定稿），
# 草稿与定稿并存，保留「审阅前 / 审阅后」的可追溯证据。
DRAFT_SUFFIX = "_draft"


def save_file(data: str, filename: str, key: str | None = None) -> str:
    """保存 Markdown 产物，返回文件路径。

    `key` 是幂等键（调用方传会话 id）：传入后文件名稳定为 `{类型}_{key}.md`，
    于是「同一会话重复收尾」——重试、恢复后再次 finish/confirm——只会覆盖同一个文件，
    不会在 Agent_output 里堆出一串内容相同的副本。不传 key 时保持原有的时间戳命名。

    刻意**不用**「临时文件 + os.replace 原子改名」：本机安全过滤驱动禁止 D 盘
    MoveFile（Vite 预构建与 Maven 资源拷贝都因此踩过坑），改名式写入在这里反而更不可靠。
    """
    safe = re.sub(r"[^\w\u4e00-\u9fa5\-]+", "_", filename).strip("_") or "output"
    if key:
        stable = re.sub(r"[^\w\u4e00-\u9fa5\-]+", "_", key).strip("_") or "session"
        file_path = config.OUTPUT_DIR / f"{safe}_{stable}.md"
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = config.OUTPUT_DIR / f"{safe}_{timestamp}.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data or "")
    logger.info("产物已保存 | %s", file_path.name)
    return str(file_path)


def list_outputs(limit: int = 100) -> list[dict]:
    """列出产物，按时间倒序。

    草稿文件名形如 `Resume_draft_20250101120000.md`，这里剥离 `_draft` 后缀再查
    类型表并标注「（草稿）」，否则会退化成查不到类型的通用「产物」。
    """
    items: list[dict] = []
    for p in config.OUTPUT_DIR.glob("*.md"):
        stat = p.stat()
        ftype = p.name.rsplit("_", 1)[0]
        is_draft = ftype.endswith(DRAFT_SUFFIX)
        base_type = ftype[: -len(DRAFT_SUFFIX)] if is_draft else ftype
        label, icon = TYPE_LABELS.get(base_type, ("产物", "📝"))
        if is_draft:
            label = f"{label}（草稿）"
        items.append(
            {
                "name": p.name,
                "path": str(p),
                "type": ftype,
                "label": label,
                "icon": icon,
                "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%m-%d %H:%M"),
                "mtime_ts": stat.st_mtime,
                "size": stat.st_size,
            }
        )
    items.sort(key=lambda x: x["mtime_ts"], reverse=True)
    return items[:limit]


def read_output(path: str) -> str:
    """读取产物内容。"""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取产物失败：%s", exc)
        return f"（读取失败：{exc}）"


def delete_output(path: str) -> bool:
    """删除产物，限制只能删除输出目录内的文件。"""
    try:
        target = Path(path).resolve()
        if config.OUTPUT_DIR.resolve() not in target.parents:
            return False
        target.unlink(missing_ok=True)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("删除产物失败：%s", exc)
        return False


def latest_output(ftype: str) -> str | None:
    """获取某类型的最新产物路径。"""
    for item in list_outputs():
        if item["type"] == ftype:
            return item["path"]
    return None
