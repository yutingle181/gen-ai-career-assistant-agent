"""检查锁文件在 **Linux** 上的依赖闭包是否完整（哈希模式失败的前置拦截）。

为什么需要这个脚本（2026-09-21 实际踩到）：
    本项目在 Windows 上开发、用 ``pip-compile`` 生成 ``requirements*.lock.txt``，而 CI 与镜像构建都在 Linux。
    只要某个依赖**不支持 Windows**，pip-compile 在 Windows 上解析时会把它**整条略过**——
    锁里既没有它的钉版行、也没有哈希；到了 Linux，它的父依赖仍然要求它，而文件里一旦出现 ``--hash``，
    pip 就**自动进入哈希模式**（不许临场补装），于是直接失败：

        ERROR: In --require-hashes mode, all requirements must have their versions pinned with ==.
        These do not: uvloop>=0.15.1 (from uvicorn[standard]==0.52.4 -> -r requirements-dev.lock.txt)

    注意报错里括号中的行号是**父依赖**的位置，不是缺失依赖的位置，很容易看错方向。
    而且这个问题**在 Windows 上装同一份锁完全正常**，只能靠 Linux 侧或本脚本提前发现。

做法：用 PyPI 元数据把锁里每个包（含 ``pkg[extra]``）在 Linux 环境下的依赖集展开，
      与锁里已有的包名做差集 —— 差集非空即表示「Linux 会要求但锁里没有」，需要补钉版+哈希。

用法：
    python scripts/check_lock_platform_deps.py requirements.lock.txt requirements-dev.lock.txt
退出码：0 = 全部齐全；1 = 存在缺口（会把缺口打印出来）。
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

#: 需求行：``name[extra1,extra2]==version``（以 ``==`` 为界，锁里必须全量钉版）
PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?:\[(?P<extras>[^\]]*)\])?==(?P<ver>[^\s;\\]+)")

#: 用来求值环境标记的 Linux 环境（与 CI 的 ubuntu-latest + setup-python 3.12 对齐）
LINUX_ENV = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "implementation_name": "cpython",
    "os_name": "posix",
    "python_version": "3.12",
    "python_full_version": "3.12.0",
}

_FETCH_TRIES = 3


def fetch_requires(pkg: str, ver: str) -> list[str]:
    """取某个版本的 ``requires_dist``（带重试：PyPI 偶发连接中断不该让检查误报）。"""
    last: Exception | None = None
    for _attempt in range(_FETCH_TRIES):
        try:
            request = urllib.request.Request(
                f"https://pypi.org/pypi/{pkg}/{ver}/json",
                headers={"User-Agent": "check-lock-platform-deps/1.0"},
            )
            with urllib.request.urlopen(request, timeout=45) as resp:
                return json.load(resp).get("requires_dist") or []
        except Exception as exc:  # noqa: BLE001 - 网络问题统一重试
            last = exc
    raise RuntimeError(f"{pkg}=={ver} 取元数据失败：{last}")


def linux_requirements(pkg: str, ver: str, extras: list[str]) -> set[str]:
    """该包在 Linux 上需要的依赖包名集合（含基础依赖与指定 extras 的依赖）。"""
    needed: set[str] = set()
    for raw in fetch_requires(pkg, ver):
        try:
            requirement = Requirement(raw)
        except Exception:  # noqa: BLE001 - 元数据里的历史语法问题，跳过即可
            continue
        if requirement.marker is None:
            needed.add(canonicalize_name(requirement.name))
            continue
        # 逐个 extra 求值（空字符串 = 基础依赖）；任一命中即算需要
        for extra in ("", *extras):
            try:
                if requirement.marker.evaluate(dict(LINUX_ENV, extra=extra)):
                    needed.add(canonicalize_name(requirement.name))
                    break
            except Exception:  # noqa: BLE001 - 标记含未知变量时保守跳过
                continue
    return needed


def parse_lock(path: pathlib.Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    pins: dict[str, str] = {}
    extras_map: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PIN.match(line)
        if not match:
            continue
        key = canonicalize_name(match.group("name"))
        pins[key] = match.group("ver")
        raw_extras = match.group("extras")
        if raw_extras:
            extras_map[key] = [e.strip() for e in raw_extras.split(",") if e.strip()]
    return pins, extras_map


def check(path: pathlib.Path) -> dict[str, list[str]]:
    pins, extras_map = parse_lock(path)
    print(f"=== {path.name}：{len(pins)} 个包（带 extra 的 {len(extras_map)} 个）===")
    if not pins:
        print("  ✗ 没解析到任何钉版行，文件格式或路径有问题")
        return {"<file>": ["<未解析到钉版行>"]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(
            pool.map(
                lambda kv: (kv[0], linux_requirements(kv[0], kv[1], extras_map.get(kv[0], []))),
                pins.items(),
            )
        )

    gaps: dict[str, list[str]] = {}
    for pkg, needed in results.items():
        missing = sorted(name for name in needed if name not in pins)
        if missing:
            gaps[pkg] = missing
    return gaps


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    failed = False
    for raw in argv[1:]:
        path = pathlib.Path(raw)
        if not path.exists():
            print(f"✗ 找不到文件：{path}")
            failed = True
            continue
        gaps = check(path)
        if not gaps:
            print("  ✓ Linux 依赖闭包完整（哈希模式不会因缺依赖失败）\n")
            continue
        failed = True
        print("  ✗ 存在缺口：下列依赖在 Linux 上会被要求，却没有出现在锁里（补钉版 + 哈希）：")
        for pkg, missing in sorted(gaps.items()):
            print(f"      {pkg} -> {', '.join(missing)}")
        print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
