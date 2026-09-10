"""Markdown 产物存储：落盘、列表、读取、删除与越权防护。

重点覆盖 `delete_output` 的目录越权防护——只能删除输出目录内的文件。
"""

from __future__ import annotations

import pytest

from src import config, storage


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    """把产物目录重定向到临时目录，避免污染真实 Agent_output/。"""
    d = tmp_path / "Agent_output"
    d.mkdir()
    monkeypatch.setattr(config, "OUTPUT_DIR", d)
    return d


def test_save_file_writes_utf8(out_dir):
    path = storage.save_file("# 标题\n内容", "Tutorial")
    assert path.endswith(".md")
    assert "Tutorial" in path.split("\\")[-1].split("/")[-1]
    with open(path, encoding="utf-8") as f:
        assert f.read() == "# 标题\n内容"


def test_save_file_sanitizes_filename(out_dir):
    path = storage.save_file("x", "bad/name:中文!")
    name = path.split("\\")[-1].split("/")[-1]
    assert "/" not in name and ":" not in name


def test_list_outputs_returns_metadata(out_dir):
    storage.save_file("内容", "Tutorial")
    items = storage.list_outputs()
    assert items
    assert items[0]["type"] == "Tutorial"
    assert items[0]["label"] == "教程"
    assert items[0]["size"] > 0


def test_read_output_missing_file_does_not_raise(out_dir):
    assert "读取失败" in storage.read_output(str(out_dir / "nope.md"))


def test_delete_output_inside_dir(out_dir):
    path = storage.save_file("内容", "Resume")
    assert storage.delete_output(path) is True
    assert not storage.list_outputs()


def test_delete_output_rejects_path_outside_dir(out_dir, tmp_path):
    outside = tmp_path / "secret.md"
    outside.write_text("不能删", encoding="utf-8")
    assert storage.delete_output(str(outside)) is False
    assert outside.exists()


def test_latest_output(out_dir):
    assert storage.latest_output("Tutorial") is None
    path = storage.save_file("内容", "Tutorial")
    assert storage.latest_output("Tutorial") == path
