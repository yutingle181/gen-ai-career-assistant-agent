"""pytest 全局配置：确保仓库根目录可导入。

这样无论用 `pytest`、`python -m pytest` 还是 IDE 直接跑单个文件，
`import src...` 都能正常工作（不再依赖各测试文件里的 sys.path 补丁）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
