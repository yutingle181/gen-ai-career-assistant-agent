"""图级 checkpoint 的获取与降级（P1-4 可恢复性）。

**为什么自己做一层工厂**：`langgraph-checkpoint-sqlite` 是可选依赖。
如果直接 `from langgraph.checkpoint.sqlite import SqliteSaver`，没装包的环境会
在 import 阶段就把服务拖崩；如果干脆不接 checkpointer，「重试不重复执行工具」
这个能力又会悄悄消失。所以这里把「用哪个 saver」显式化，并允许降级：

1. `SqliteSaver`（`langgraph-checkpoint-sqlite` 已装）→ **跨进程**持久化，进程重启后仍可续跑；
2. `InMemorySaver`（`langgraph-checkpoint` 自带）→ 进程内有效：同一轮重试命中 checkpoint，
   不会重复调用模型与工具；
3. 两者都不可用 → 返回 `None`，调用方退化为「不带 checkpointer」的老行为（零回归、不报错）。

恢复语义的边界（面试可讲）：这里解决的是**单轮工具闭环**的续跑；
「多轮对话跨重启继续」由会话级重建承担（`src/api/deps.py: restore_session`）——
两者职责不同，刻意不互相假装。
"""

from __future__ import annotations

from typing import Any

from .. import config
from ..logging_setup import get_logger

logger = get_logger(__name__)

#: 已构建的 saver（进程内单例）。None 表示「尝试过且不可用」，与「尚未尝试」区分。
_SAVER: Any = None
_BACKEND: str = ""
_READY = False


def _build_saver() -> tuple[Any, str]:
    """按「sqlite → memory → 无」的顺序尝试构建 saver。"""
    if not config.ENABLE_CHECKPOINT:
        return None, "disabled"

    if config.CHECKPOINT_DB_PATH:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            saver = SqliteSaver.from_conn_string(config.CHECKPOINT_DB_PATH)
            # from_conn_string 在不同版本返回上下文管理器或 saver 本身，两种都兼容
            saver = getattr(saver, "__enter__", lambda: saver)()
            return saver, "sqlite"
        except Exception as exc:  # noqa: BLE001
            logger.info("SqliteSaver 不可用，降级为内存 checkpointer：%s", exc)

    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver(), "memory"
    except Exception as exc:  # noqa: BLE001
        logger.info("内存 checkpointer 不可用，本轮不启用 checkpoint：%s", exc)
        return None, "none"


def get_checkpointer() -> tuple[Any, str]:
    """获取 (saver, 后端名)。后端名取值：sqlite / memory / none / disabled。

    首次调用时构建并缓存；任何异常都不会外抛——可恢复性不该成为启动的阻塞点。
    """
    global _SAVER, _BACKEND, _READY
    if not _READY:
        _READY = True
        _SAVER, _BACKEND = _build_saver()
        if _SAVER is None:
            logger.info("图级 checkpoint 未启用（backend=%s）", _BACKEND)
        else:
            logger.info("图级 checkpoint 已启用 | backend=%s", _BACKEND)
    return _SAVER, _BACKEND


def reset_checkpointer() -> None:
    """清空缓存（配置热更新与测试用）。"""
    global _SAVER, _BACKEND, _READY
    _SAVER, _BACKEND, _READY = None, "", False


def backend_name() -> str:
    """当前 checkpoint 后端名（供 /health 与日志展示）。"""
    return get_checkpointer()[1]
