"""pytest 全局配置：确保仓库根目录可导入，并隔离进程内共享状态。

这样无论用 `pytest`、`python -m pytest` 还是 IDE 直接跑单个文件，
`import src...` 都能正常工作（不再依赖各测试文件里的 sys.path 补丁）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _reset_process_state():
    """每个用例前后重置进程内共享状态（工具熔断器 / 指标计数器）。

    这两个东西在生产里是**有意做成进程内单例**的，但在测试里会跨用例泄漏：
    例如某个用例连续制造检索失败把熔断器打开，后面「期望真实调用」的用例
    就会莫名拿到空结果（表现为随机顺序下才失败，极难排查）。
    统一在 conftest 里重置，比在每个测试文件里各写一遍更不容易漏。
    """
    from src import metrics
    from src.tools import get_breaker

    get_breaker().reset()
    metrics.reset()
    yield
    get_breaker().reset()
    metrics.reset()
