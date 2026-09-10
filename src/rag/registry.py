"""多知识库管理：创建、加载、切换、删除、统计。"""

from __future__ import annotations

import builtins
import shutil

from .. import config
from ..logging_setup import get_logger
from .pipeline import RAGPipeline

logger = get_logger(__name__)

DEFAULT_KB = "default"


class KBRegistry:
    """知识库注册表，索引目录下一级子目录即一个知识库。"""

    def __init__(self) -> None:
        self._pipelines: dict[str, RAGPipeline] = {}
        self._scan()

    def _scan(self) -> None:
        if not config.INDEX_DIR.exists():
            return
        for d in sorted(config.INDEX_DIR.iterdir()):
            if d.is_dir() and (d / "index.faiss").exists():
                self._pipelines[d.name] = RAGPipeline(d.name, d)
        if self._pipelines:
            logger.info("已加载知识库：%s", ", ".join(self._pipelines))

    def names(self) -> builtins.list[str]:
        return sorted(self._pipelines)

    def get(self, name: str) -> RAGPipeline | None:
        return self._pipelines.get(name)

    def get_or_create(self, name: str = DEFAULT_KB) -> RAGPipeline:
        p = self._pipelines.get(name)
        if p is None:
            p = RAGPipeline(name)
            self._pipelines[name] = p
            logger.info("创建知识库：%s", name)
        return p

    def create(self, name: str) -> RAGPipeline:
        return self.get_or_create(name)

    def delete(self, name: str) -> bool:
        p = self._pipelines.pop(name, None)
        if p is None:
            return False
        try:
            shutil.rmtree(p.index_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("删除知识库目录失败：%s", exc)
        return True

    def list(self) -> builtins.list[dict]:
        return [p.stats() for p in self._pipelines.values()]

    def ensure_default(self) -> RAGPipeline:
        """确保存在一个默认知识库，便于首次启动即可用。"""
        if not self._pipelines:
            return self.get_or_create(DEFAULT_KB)
        return self._pipelines.get(DEFAULT_KB) or self._pipelines[sorted(self._pipelines)[0]]


_registry: KBRegistry | None = None


def get_registry() -> KBRegistry:
    global _registry
    if _registry is None:
        _registry = KBRegistry()
    return _registry
