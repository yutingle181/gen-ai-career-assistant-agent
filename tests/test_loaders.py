"""文档解析：按后缀分发、失败降级、批量加载。

对应 JD 职责 3「文档解析」。这里只覆盖不需要外部解析器的分支，
PDF / DOCX / HTML 在依赖缺失时自动跳过。
"""

from __future__ import annotations

import pytest

from src.rag.loaders import load_dir, load_file, load_paths


def test_load_text_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("这是纯文本内容", encoding="utf-8")
    docs = load_file(p)
    assert len(docs) == 1
    assert docs[0].source == "doc.txt"
    assert "纯文本" in docs[0].text


def test_load_markdown_file(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# 标题\n内容", encoding="utf-8")
    docs = load_file(p)
    assert len(docs) == 1 and "标题" in docs[0].text


def test_unsupported_suffix_returns_empty(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"\x00\x01")
    assert load_file(p) == []


def test_missing_file_returns_empty(tmp_path):
    """解析异常必须降级为空列表，而不是抛给上层。"""
    assert load_file(tmp_path / "nope.txt") == []


def test_load_dir_only_picks_supported(tmp_path):
    (tmp_path / "a.txt").write_text("内容", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00")
    docs = load_dir(tmp_path)
    assert len(docs) == 1 and docs[0].source == "a.txt"


def test_load_dir_missing_returns_empty(tmp_path):
    assert load_dir(tmp_path / "nope") == []


def test_load_paths_mixed(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.md").write_text("目录内", encoding="utf-8")
    single = tmp_path / "single.txt"
    single.write_text("单文件", encoding="utf-8")
    docs = load_paths([tmp_path / "sub", single])
    assert {d.source for d in docs} == {"inner.md", "single.txt"}


def test_load_html(tmp_path):
    pytest.importorskip("bs4")
    p = tmp_path / "page.html"
    p.write_text(
        "<html><body><script>alert(1)</script><h1>标题</h1><p>正文</p></body></html>",
        encoding="utf-8",
    )
    docs = load_file(p)
    assert len(docs) == 1
    assert "alert" not in docs[0].text  # script 标签应被剔除
    assert "正文" in docs[0].text
