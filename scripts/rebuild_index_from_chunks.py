"""用当前 Embedding 模型重建某个知识库的向量索引（复用已切好的片段，不重新解析源文档）。

为什么需要这个脚本（2026-10-15 实测踩坑）：
换 Embedding 模型或维度后（如云端 text-embedding-v3 1024 维 → 本地 BGE 512 维），
旧 faiss 索引**维度不匹配却仍被判为"就绪"**——`RAGPipeline.ready` 只看「有没有片段」，
于是问题一直潜伏到真正查询时才在 faiss 里炸出来，报错信息还与根因（模型换了）毫无关联。
而此时源文档往往已经不在手边（例如知识库来自上传目录），`ingest` 重建这条路走不通。

本脚本的定位：直接读回 `data/index/<kb>/chunks.pkl` 里**已经切好的片段**，
用当前 Embedding 重新向量化并覆盖写回索引 —— 不重新解析文档、不改动切分参数，
因此检索结果的变化只来自向量模型本身，对照实验才有意义。

用法：
    .venv-ci\\Scripts\\python.exe scripts/rebuild_index_from_chunks.py --kb data1
    .venv-ci\\Scripts\\python.exe scripts/rebuild_index_from_chunks.py --kb data1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.embeddings import get_embedder  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402
from src.rag.types import Chunk  # noqa: E402,F401  # 保证 pickle 能反序列化 Chunk
from src.rag.vectorstore import FaissStore  # noqa: E402

logger = get_logger(__name__)


def _load_meta(index_dir: Path) -> dict:
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("meta.json 读取失败（忽略）：%s", exc)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="从已存片段重建向量索引（换 Embedding 模型后使用）")
    parser.add_argument("--kb", required=True, help="知识库名（对应 data/index/<kb>/）")
    parser.add_argument("--index-dir", default="", help="索引目录，缺省用 data/index/<kb>")
    parser.add_argument("--dry-run", action="store_true", help="只打印现状（片段数/旧维度/新维度），不写盘")
    args = parser.parse_args()

    index_dir = Path(args.index_dir) if args.index_dir else config.INDEX_DIR / args.kb
    chunks_path = index_dir / "chunks.pkl"
    if not chunks_path.exists():
        print(f"[XX] 找不到 {chunks_path}；该知识库可能从未建库，或索引目录不对")
        return 2

    with open(chunks_path, "rb") as fh:
        chunks: list[Chunk] = pickle.load(fh)
    if not chunks:
        print(f"[XX] {chunks_path} 里没有片段，无法重建")
        return 2

    meta = _load_meta(index_dir)
    old_dim = meta.get("dim", "未知")
    print(f"知识库        : {args.kb}")
    print(f"索引目录      : {index_dir}")
    print(f"已存片段      : {len(chunks)} 个（沿用原切分，不重新解析源文档）")
    print(f"旧索引维度    : {old_dim}")

    embedder = get_embedder()
    model = config.EMBEDDING_MODEL if embedder.name == "api" else config.LOCAL_EMBEDDING_MODEL
    print(f"当前向量模型  : {embedder.name}/{model}")

    if args.dry_run:
        print(f"新向量维度    : {embedder.dim}（探测得到，未写盘）")
        if isinstance(old_dim, int) and old_dim != embedder.dim:
            print("→ 维度不一致，确认无误后去掉 --dry-run 执行重建")
        else:
            print("→ 维度一致，其实无需重建（仍可执行，覆盖写回不影响结果）")
        return 0

    started = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    new_dim = len(vectors[0])
    if isinstance(old_dim, int) and old_dim != new_dim:
        print(f"维度变化      : {old_dim} → {new_dim}（这正是必须重建的原因）")

    store = FaissStore(new_dim)
    store.add(chunks, vectors)
    store.save(index_dir)
    print(f"重建完成      : {len(chunks)} 片段 / {new_dim} 维 / {time.perf_counter() - started:.1f}s")
    print("提示          : 若报告里的检索指标与之前不可比，请确认对比的是同一批片段与同一切分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
