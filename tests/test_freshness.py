"""数据时效（P0-3 收尾）：工具必须把「过没过期」算好，而不是只印一个日期。

要点：模型没有时钟、也不擅长日期减法。只给「文档时间：2023-05-01」，
它照样会把三年前的数字当现状；给出「距今 1236 天，可能已过期」才拦得住。
本文件全部离线、不依赖真实时钟（时间一律显式传入或取相对值）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src import config, freshness, metrics, tools


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(config, "FRESHNESS_STALE_DAYS", 365)
    monkeypatch.setattr(config, "FRESHNESS_SPAN_DAYS", 180)
    metrics.reset()
    yield
    metrics.reset()


NOW = datetime(2026, 9, 19, 12, 0, 0)


# ------------------------------------------------------------------ 解析
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2023-05-01", datetime(2023, 5, 1)),
        ("2023-05-01 10:20:30", datetime(2023, 5, 1, 10, 20, 30)),
        ("2023-05-01T10:20", datetime(2023, 5, 1, 10, 20)),
        ("2023/05/01", datetime(2023, 5, 1)),
        ("20230501", datetime(2023, 5, 1)),
        (datetime(2024, 1, 2, 3, 4), datetime(2024, 1, 2, 3, 4)),
    ],
)
def test_parse_time_accepts_common_writings(value, expected):
    assert freshness.parse_time(value) == expected


def test_parse_time_accepts_epoch_seconds_and_millis():
    assert freshness.parse_time(1682900000) == datetime.fromtimestamp(1682900000)
    assert freshness.parse_time(1682900000000) == datetime.fromtimestamp(1682900000)


@pytest.mark.parametrize("value", [None, "", "未知", "去年", "近期", "2023-13-45"])
def test_parse_time_refuses_to_guess(value):
    """解析不了就说不知道 —— 猜错比说「未知」更危险。"""
    assert freshness.parse_time(value) is None


def test_parse_time_does_not_mistake_snowflake_id_for_date():
    """19 位雪花 id 不能被当成时间戳（否则会渲染出一个荒唐的年份）。"""
    assert freshness.parse_time("2100849832599060481") is None


# ------------------------------------------------------------------ 标注
def test_annotate_flags_stale_document():
    label, suspect = freshness.annotate("2023-05-01", now=NOW)

    assert suspect is True
    assert "2023-05-01" in label
    assert f"距今 {(NOW - datetime(2023, 5, 1)).days} 天" in label
    assert "可能已过期" in label


def test_annotate_keeps_fresh_document_clean():
    recent = (NOW - timedelta(days=30)).strftime("%Y-%m-%d")
    label, suspect = freshness.annotate(recent, now=NOW)

    assert suspect is False
    assert "距今 30 天" in label
    assert "可能已过期" not in label


def test_annotate_unknown_is_declared_not_silently_blank():
    label, suspect = freshness.annotate("未知", now=NOW)

    assert suspect is False  # 未知 ≠ 过期，但同样不能默认它最新
    assert label == freshness.UNKNOWN_LABEL
    assert "无法判断时效" in label


def test_annotate_treats_future_time_as_untrustworthy():
    """未来时间同样是「不能当当下事实用」，必须标出来而不是照单全收。"""
    future = (NOW + timedelta(days=10)).strftime("%Y-%m-%d")
    label, suspect = freshness.annotate(future, now=NOW)

    assert suspect is True
    assert "疑似时间异常" in label


def test_stale_threshold_is_configurable(monkeypatch):
    monkeypatch.setattr(config, "FRESHNESS_STALE_DAYS", 10)
    recent = (NOW - timedelta(days=30)).strftime("%Y-%m-%d")

    _label, suspect = freshness.annotate(recent, now=NOW)

    assert suspect is True


def test_age_days_returns_none_when_unparsable():
    assert freshness.age_days("未知", now=NOW) is None
    assert freshness.age_days("2023-05-01", now=NOW) == (NOW - datetime(2023, 5, 1)).days


# ------------------------------------------------------------------ 多版本跨度
def test_span_note_silent_when_versions_are_close():
    values = ["2026-08-01", "2026-08-20", "2026-09-01"]
    assert freshness.span_note(values) == ""


def test_span_note_surfaces_wide_version_span():
    note = freshness.span_note(["2023-01-01", "2026-09-01"])

    assert "时间跨度" in note and "不同版本" in note
    assert "2023-01-01" in note and "2026-09-01" in note


def test_span_note_needs_at_least_two_known_times():
    assert freshness.span_note(["2023-01-01", "未知"]) == ""


def test_audit_counts_stale_unknown_and_fresh():
    values = ["2023-05-01", "未知", (NOW - timedelta(days=5)).strftime("%Y-%m-%d")]

    stats = freshness.audit(values, now=NOW)

    assert stats == {"total": 3, "stale": 1, "unknown": 1, "fresh": 1}


# ------------------------------------------------------------------ 工具接入
def _kb_tool(monkeypatch, chunks: list):
    class _Pipeline:
        def retrieve(self, query, cfg=None):
            return list(chunks)

    class _Registry:
        def get(self, name):
            return _Pipeline()

    import src.rag.registry as registry

    monkeypatch.setattr(registry, "get_registry", lambda: _Registry())
    return tools._knowledge_search_tool("kb")


def _chunk(source="doc.md", updated="2023-05-01", text="内容"):
    return SimpleNamespace(source=source, page=1, text=text, meta={"updated_at": updated})


def test_knowledge_tool_marks_stale_result(monkeypatch):
    """片段过期时，工具输出里必须带「可能已过期」，并计入指标。"""
    out = _kb_tool(monkeypatch, [_chunk()]).invoke({"query": "问题"})

    assert "文档时间：" in out and "可能已过期" in out
    assert metrics.counter("tool.result.stale") == 1


def test_knowledge_tool_marks_unknown_time(monkeypatch):
    chunk = SimpleNamespace(source="doc.md", page=1, text="内容", meta={})
    out = _kb_tool(monkeypatch, [chunk]).invoke({"query": "问题"})

    assert "无法判断时效" in out
    assert metrics.counter("tool.result.stale") == 0


def test_knowledge_tool_surfaces_version_span(monkeypatch):
    """同一来源出现新旧两个版本时，工具摆出跨度而不是替模型合并结论。"""
    chunks = [_chunk(updated="2023-01-01"), _chunk(updated="2026-09-01")]
    out = _kb_tool(monkeypatch, chunks).invoke({"query": "问题"})

    assert "时间跨度" in out and "不同版本" in out


def test_fresh_chunks_have_no_stale_noise(monkeypatch):
    fresh = datetime.now().strftime("%Y-%m-%d")
    out = _kb_tool(monkeypatch, [_chunk(updated=fresh)]).invoke({"query": "问题"})

    assert "可能已过期" not in out and "时间跨度" not in out
    assert metrics.counter("tool.result.stale") == 0


# ------------------------------------------------------------------ 数据侧健康度
def test_chunk_time_extraction(tmp_path):
    """取时间的实现只有一份（freshness.chunk_time），工具与统计共用。"""
    from_meta = SimpleNamespace(source="(不存在).md", meta={"updated_at": "2025-01-02 10:30:00"})
    assert freshness.chunk_time(from_meta) == "2025-01-02 10:30"

    doc = tmp_path / "kb.md"
    doc.write_text("内容", encoding="utf-8")
    assert freshness.chunk_time(SimpleNamespace(source=str(doc), meta={}))[:4] == str(
        datetime.now().year
    )

    assert freshness.chunk_time(SimpleNamespace(source="缺失.md", meta={})) == "未知"


def test_pipeline_stats_reports_freshness(tmp_path):
    """知识库统计要给出数据侧新鲜度——过期资料是数据问题，不该只靠模型声明。"""
    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline("freshness-test", index_dir=tmp_path / "idx")
    pipeline.chunks = [
        SimpleNamespace(source="老文档.md", meta={"updated_at": "2019-01-01"}),
        SimpleNamespace(source="无时间.md", meta={}),
    ]

    stats = pipeline.stats()

    assert stats["freshness"]["total"] == 2
    assert stats["freshness"]["stale"] == 1
    assert stats["freshness"]["unknown"] == 1


def test_report_prints_freshness_line(tmp_path):
    """报告开头必须能看到过期片段数（否则「该更新文档」永远不会成为行动项）。"""
    from src.eval.report import render_report
    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline("freshness-report", index_dir=tmp_path / "idx")
    pipeline.chunks = [SimpleNamespace(source="老文档.md", meta={"updated_at": "2019-01-01"})]

    report = render_report([], pipeline=pipeline, k=5)

    assert "资料新鲜度" in report and "过期 1" in report


# ------------------------------------------------------------------ 检索层必须带出 meta
class _FakeEmbedder:
    """确定性伪 Embedding（本文件不联网，也不需要真实向量质量）。"""

    name = "fake"

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str):
        vec = [0.0] * self._dim
        for ch in (text or ""):
            vec[ord(ch) % self._dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


def test_retrieval_carries_chunk_meta(tmp_path, monkeypatch):
    """检索层必须把片段 meta 带出来——否则「资料是什么时候的」在工具层永远是「未知」。

    这是端到端跑真实链路才暴露的问题：`RetrievedChunk` 当初没有 meta 字段，
    入库时间在检索出口被丢掉，工具只能靠「源文件路径还在不在」反推，
    文件一被移走 / 清理，时效就永久变成「未知」（data1 里 3/10 片段就是这样）。
    """
    import os

    import src.embeddings as emb
    import src.rag.pipeline as pipe
    from src.rag.pipeline import RAGPipeline, RetrievalConfig

    monkeypatch.setattr(emb, "get_embedder", lambda force=False: _FakeEmbedder())
    monkeypatch.setattr(pipe, "get_embedder", emb.get_embedder)

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    doc = src_dir / "policy.md"
    doc.write_text("出差住宿标准：一线城市每晚 500 元。", encoding="utf-8")
    old = datetime(2019, 1, 1, 9, 0).timestamp()
    os.utime(doc, (old, old))

    pipeline = RAGPipeline("freshness-meta", index_dir=tmp_path / "idx")
    pipeline.ingest([str(src_dir)])
    chunks = pipeline.retrieve(
        "出差住宿标准", RetrievalConfig(top_k=3, reranker="none"), use_cache=False
    )

    assert chunks, "检索应至少返回一条片段"
    assert chunks[0].meta.get("updated_at"), "检索层丢掉了片段 meta"
    label, suspect = freshness.annotate(freshness.chunk_time(chunks[0]))
    assert suspect is True and "可能已过期" in label


# ------------------------------------------------------------------ Prompt 契约
def test_persona_carries_freshness_rule():
    """规则必须与工具的标注对得上：工具给了什么词，prompt 就得要求声明什么。"""
    from src.prompts.personas import get_persona

    persona = get_persona("qa")

    assert "距今天数" in persona
    assert "可能已过期" in persona
    assert "时间跨度" in persona
    assert "工具降级" in persona
