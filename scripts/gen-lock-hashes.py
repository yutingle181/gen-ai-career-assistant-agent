"""为 pip-compile 生成的锁文件补充 PyPI 官方哈希（供 `pip install --require-hashes` 使用）。

为什么不用 `pip-tools --generate-hashes`：
  1. 它会**下载**候选文件来算哈希（本项目单平台编译时可能只覆盖当前平台的轮子），
     全平台下载动辄数 GB —— 本机 C 盘空间不足时会直接失败；
  2. 本脚本只查询 PyPI JSON 元数据（**零下载**），因此可以一次覆盖某版本**已发布的全部文件**
     （Windows 轮子 + manylinux/musllinux 轮子 + sdist），本地开发与 Linux 容器/CI 都能校验通过。

用法（在仓库根目录执行，改完 requirements*.txt 并 pip-compile 之后）：
    python scripts/gen-lock-hashes.py requirements.lock.txt requirements-dev.lock.txt

行为：就地重写锁文件——每个 `name==version` 行下方追加 `--hash=sha256:...` 行（缩进 4 空格），
      其它内容（注释、`# via` 说明）原样保留；任何取不到哈希的包会报错退出，不会写出半成品。
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# 形如 `name==1.2.3`、`name[extra]==1.2.3`、可选 `; marker`
REQ_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]*\])?==(?P<version>[^\s;]+)(?P<marker>\s*;.*)?$")


def fetch_hashes(name: str, version: str) -> tuple[str, list[str]]:
    """取某版本在 PyPI 上全部文件的 sha256（含跨平台轮子与 sdist）。网络抖动自动重试。"""
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    last: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gen-lock-hashes/1.0"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.load(resp)
            hashes = {
                f["digests"]["sha256"]
                for f in data.get("urls", [])
                if (f.get("digests") or {}).get("sha256")
            }
            return f"{name}=={version}", sorted(hashes)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"!! 取元数据失败 {name}=={version}（重试 3 次）：{last}")


def augment(path: pathlib.Path) -> tuple[int, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    pinned: list[tuple[str, str]] = []
    for line in lines:
        m = REQ_RE.match(line.strip())
        if m:
            pinned.append((m.group("name"), m.group("version")))

    if not pinned:
        raise SystemExit(f"!! {path}: 没解析到任何 name==version 行，疑似文件格式不对")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(pool.map(lambda nv: fetch_hashes(*nv), pinned))

    missing = [f"{n}=={v}" for n, v in pinned if not results.get(f"{n}=={v}")]
    if missing:
        raise SystemExit("!! 以下包没取到哈希，已中止：" + ", ".join(missing))

    out: list[str] = []
    for line in lines:
        m = REQ_RE.match(line.strip())
        if not m:
            out.append(line)
            continue
        # 关键：必须用反斜杠续行（与 pip-compile --generate-hashes 的输出格式一致），
        # 否则 pip 会把每个 --hash 行当成"没有需求的行"而忽略掉。
        hashes = results[f"{m.group('name')}=={m.group('version')}"]
        out.append(f"{line} \\")
        for i, h in enumerate(hashes):
            suffix = " \\" if i < len(hashes) - 1 else ""
            out.append(f"    --hash=sha256:{h}{suffix}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(pinned), [f"{n}=={v}" for n, v in pinned]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        path = pathlib.Path(arg)
        if not path.exists():
            print(f"!! 文件不存在：{path}")
            return 1
        count, names = augment(path)
        print(f"[OK] {path}: 为 {count} 个包补齐哈希")
        if len(names) <= 5:
            print("     " + ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
