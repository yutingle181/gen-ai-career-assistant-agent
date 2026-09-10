"""评测体系测试：用确定性伪 Embedding 跑通指标与报告生成（不依赖真实 API）。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import config
from src.eval import (
    build_offline_records,
    default_experiments,
    evaluate_retrieval,
    recall_at_k,
    run_all,
    runner,
)
from src.eval.dataset import _parse_qa, load_records, save_records
from src.eval.metrics import _parse_claims, judge_hallucination
from src.eval.report import render_report
from src.models import EvalRecord
from src.rag.pipeline import RAGPipeline, RetrievalConfig


class _FakeEmbedder:
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


def test_metrics_pure_functions():
    assert recall_at_k(["a", "b", "c"], ["b"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["z"], k=3) == 0.0
    assert recall_at_k(["a", "b", "z"], ["z"], k=2) == 0.0
    recall, mrr, hr = evaluate_retrieval(
        [(["a", "b"], ["b"]), (["x", "y"], ["z"]), (["m"], ["m"])], k=2
    )
    assert abs(recall - 2 / 3) < 1e-6, recall
    assert abs(mrr - (0.5 + 0 + 1.0) / 3) < 1e-6, mrr
    assert abs(hr - 2 / 3) < 1e-6
    print("[OK] Recall@K / MRR / HitRate 计算正确")


def test_eval_pipeline_and_report():
    import src.embeddings as emb
    import src.rag.pipeline as pipe

    fake = _FakeEmbedder()
    emb.get_embedder = lambda force=False: fake
    pipe.get_embedder = emb.get_embedder

    tmp = Path(tempfile.mkdtemp(prefix="eval_test_"))
    try:
        pl = RAGPipeline("eval_kb", index_dir=tmp)
        pl.ingest([str(config.SAMPLES_DIR)], chunk_size=300, chunk_overlap=50)
        assert pl.ready

        records = build_offline_records(pl)
        assert records, "离线评测集为空"

        results = run_all(pl, records, default_experiments(), k=5, with_hallucination=False)
        assert len(results) == 4
        assert all(r.sample_count > 0 for r in results)
        for r in results:
            print(
                f"    - {r.name}: Recall@5={r.recall_at_k:.3f} MRR={r.mrr:.3f} "
                f"延迟={r.avg_latency_ms}ms"
            )

        md = render_report(results, pl, k=5)
        assert "Recall@5" in md and "结论与优化建议" in md
        print("[OK] 评测跑批与报告渲染正常")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_metrics_edge_cases():
    """空输入必须返回 0 而不是除零。"""
    assert recall_at_k(["a"], [], k=5) == 0.0
    assert evaluate_retrieval([]) == (0.0, 0.0, 0.0)


def test_judge_hallucination_skips_empty_input():
    assert judge_hallucination("", []) == (0.0, 0)
    assert judge_hallucination("有回答但无召回", []) == (0.0, 0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('```json\n{"claims":[{"supported":true},{"supported":false}]}\n```', 2),
        ("模型没有返回 JSON", 0),
        ('{"claims": "格式不对"}', 0),
    ],
)
def test_parse_claims_handles_messy_output(raw, expected):
    assert len(_parse_claims(raw)) == expected


def test_parse_qa_filters_invalid_items():
    assert _parse_qa('```json\n[{"question":"q1","answer":"a1"}]\n```') == [
        {"question": "q1", "answer": "a1"}
    ]
    assert _parse_qa("没有数组") == []
    assert _parse_qa('[{"question":"","answer":"缺问题"}]') == []


def test_save_and_load_records_roundtrip(tmp_path):
    path = tmp_path / "eval.jsonl"
    save_records(
        [EvalRecord(question="q", golden_chunk_ids=["c1"], reference_answer="a")],
        path,
    )
    loaded = load_records(path)
    assert len(loaded) == 1
    assert loaded[0].question == "q"
    assert loaded[0].golden_chunk_ids == ["c1"]


def test_load_records_missing_file_returns_empty(tmp_path):
    assert load_records(tmp_path / "not_exists.jsonl") == []


def test_run_all_judges_hallucination_once(monkeypatch):
    """幻觉率默认只在基线判定一次；逐组判定会成倍放大 LLM 调用。"""
    calls: list[int] = []

    def fake_judge(answer, chunks):
        calls.append(1)
        return 0.0, 1

    monkeypatch.setattr(runner, "judge_hallucination", fake_judge)

    class _Chunk:
        def __init__(self, cid: str) -> None:
            self.chunk_id = cid
            self.text = "内容" * 100

    class _FakePipeline:
        def retrieve(self, query, cfg=None):
            return [_Chunk("c1"), _Chunk("c2")]

    records = [EvalRecord(question=f"q{i}", golden_chunk_ids=["c1"]) for i in range(3)]
    experiments = default_experiments()

    results = run_all(_FakePipeline(), records, experiments, with_hallucination=True, judge_once=True)
    assert len(results) == 4
    assert len(calls) == 3, f"judge_once 应只判定 3 次（每样本一次），实际 {len(calls)}"

    # 对照组：关闭复用时应逐组判定 4 组 × 3 条 = 12 次
    calls.clear()
    run_all(
        _FakePipeline(),
        records,
        experiments,
        with_hallucination=True,
        judge_once=False,
        max_workers=1,
    )
    assert len(calls) == 12, f"judge_once=False 应判定 12 次，实际 {len(calls)}"
    print("[OK] 幻觉判定次数：复用 3 次 vs 逐组 12 次")


def test_runner_edge_cases_and_fallbacks(monkeypatch):
    """覆盖并发与降级的边界：检索失败跳过、判定异常兜底、基线配置回退。"""
    # 可选的第 5 组（API 重排）
    assert len(default_experiments(include_api_rerank=True)) == 5

    # 空输入直接返回，不进线程池
    assert runner._map_concurrent(lambda x: x, [], 4) == []

    # 无 rrf 基线时，回退到第一个不重排的配置
    only_bm25 = [("仅 BM25", RetrievalConfig(fusion="bm25_only", reranker="none"))]
    assert runner._baseline_config(only_bm25) is only_bm25[0][1]
    # 全都带重排时，回退到第一组
    only_llm = [("仅重排", RetrievalConfig(fusion="rrf", reranker="llm"))]
    assert runner._baseline_config(only_llm) is only_llm[0][1]

    # 空片段不判定
    assert runner._judge_chunks([]) == (0.0, 0)

    class _Chunk:
        chunk_id = "c1"
        text = "内容" * 100

    # 判定抛异常时兜底为 0，不影响主流程
    def boom(answer, chunks):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "judge_hallucination", boom)
    assert runner._judge_chunks([_Chunk()]) == (0.0, 0)

    class _BoomPipeline:
        def retrieve(self, query, cfg=None):
            raise RuntimeError("检索挂了")

    rec = EvalRecord(question="q", golden_chunk_ids=["c1"])
    # 检索失败：单样本返回 None，基线判定兜底为 0
    assert runner._eval_one(_BoomPipeline(), rec, only_bm25[0][1]) is None
    assert runner._judge_one(_BoomPipeline(), rec, only_bm25[0][1]) == (0.0, 0)

    # 全部检索失败时样本数为 0，不报错也不除零
    res = runner.run_experiment(
        _BoomPipeline(), [rec], only_bm25[0][1], "失败组", with_hallucination=True, max_workers=1
    )
    assert res.sample_count == 0
    print("[OK] 评测器边界与降级分支正常")


def test_build_offline_records_skips_short_chunks():
    """过短片段不适合出题，应被跳过。"""
    pipeline = SimpleNamespace(
        chunks=[
            SimpleNamespace(chunk_id="c1", text="这是一个足够长的片段内容，可以用来生成评测问题"),
            SimpleNamespace(chunk_id="c2", text="太短"),
        ]
    )
    records = build_offline_records(pipeline)
    assert [r.golden_chunk_ids[0] for r in records] == ["c1"]


if __name__ == "__main__":
    test_metrics_pure_functions()
    test_eval_pipeline_and_report()
    print("\n全部评测测试通过")
