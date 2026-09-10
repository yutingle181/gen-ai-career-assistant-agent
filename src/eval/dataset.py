"""评测集构建：从知识库片段反向生成问答对，并落盘为 JSONL。

生成策略：用 LLM 基于每个片段生成问题（保证答案确实在库里），
再抽样人工校对后保留。字段：
{"question": str, "golden_chunk_ids": [str], "reference_answer": str}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .. import config
from ..logging_setup import get_logger, summarize
from ..models import EvalRecord
from ..prompts.rag import EVAL_GEN_PROMPT
from ..rag.pipeline import RAGPipeline

logger = get_logger(__name__)

DEFAULT_PATH = config.EVAL_DIR / "eval_set.jsonl"


def generate_from_pipeline(
    pipeline: RAGPipeline,
    per_chunk: int = 2,
    max_chunks: int = 60,
) -> list[EvalRecord]:
    """基于知识库片段生成评测问答对。"""
    from langchain_core.messages import HumanMessage

    from ..llm import llm_invoke

    records: list[EvalRecord] = []
    chunks = pipeline.chunks[:max_chunks]
    for i, chunk in enumerate(chunks, start=1):
        if len(chunk.text.strip()) < 80:  # 太短的片段不适合出题
            continue
        prompt = EVAL_GEN_PROMPT.format(n=per_chunk, chunk=chunk.text[:1500])
        try:
            raw = llm_invoke([HumanMessage(content=prompt)], temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("评测集生成失败（片段 %d）：%s", i, exc)
            continue
        for qa in _parse_qa(raw):
            records.append(
                EvalRecord(
                    question=qa["question"],
                    golden_chunk_ids=[chunk.chunk_id],
                    reference_answer=qa["answer"],
                )
            )
        logger.debug("评测集生成进度 %d/%d", i, len(chunks))

    logger.info("评测集生成完成 | %d 条 | %s", len(records), summarize("; ".join(r.question for r in records[:3])))
    return records


def _parse_qa(raw: str) -> list[dict]:
    """从模型输出中解析 JSON 数组。"""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("question") and item.get("answer"):
            out.append({"question": str(item["question"]), "answer": str(item["answer"])})
    return out


def save_records(records: list[EvalRecord], path: Path | None = None) -> str:
    """保存为 JSONL。"""
    target = Path(path or DEFAULT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        for r in records:
            # 不用 model_dump_json(ensure_ascii=...)：pydantic 2.11 起不再接受该参数，
            # 改用 json.dumps 保证中文可读且不受 pydantic 版本影响。
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
    logger.info("评测集已保存 | %s | %d 条", target, len(records))
    return str(target)


def load_records(path: Path | None = None) -> list[EvalRecord]:
    """加载 JSONL 评测集。"""
    target = Path(path or DEFAULT_PATH)
    if not target.exists():
        return []
    records = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(EvalRecord.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                logger.warning("跳过非法评测条目：%s", exc)
    return records


def build_offline_records(pipeline: RAGPipeline) -> list[EvalRecord]:
    """无 LLM 时的备用方案：从片段首句构造关键词问题。

    质量不如 LLM 生成，但保证评测流程在离线环境也能跑通。
    """
    records: list[EvalRecord] = []
    for chunk in pipeline.chunks:
        first = chunk.text.strip().split("\n")[0][:60]
        if len(first) < 10:
            continue
        records.append(
            EvalRecord(
                question=f"请说明以下内容出自哪份资料：{first}",
                golden_chunk_ids=[chunk.chunk_id],
                reference_answer=chunk.text[:200],
            )
        )
    return records
