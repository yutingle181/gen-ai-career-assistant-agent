"""P1-4 跨进程恢复：真起子进程、真让进程崩，验证「重启后续跑且不重复执行」。

为什么单独一个文件、且用子进程：
    进程内 memory saver 的复用已在 test_checkpoint_resume.py 覆盖（同轮重试不重复执行）；
    但「进程挂掉后还能续」只有在**另一个进程**里才能证明，所以这里用 subprocess 拉起
    tests/_resume_probe.py，并显式 kill 掉一个。

依赖：需要可选的 `langgraph-checkpoint-sqlite`（开发依赖里已声明）。没有就整体 skip——
保证「不装可选依赖也能跑全量测试」这条老约定不被破坏。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "langgraph.checkpoint.sqlite",
    reason="需要可选依赖 langgraph-checkpoint-sqlite（requirements-dev.txt 已声明）",
)

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "tests" / "_resume_probe.py"


def _run_probe(db: Path, thread: str, counter: Path, *, kill_after_first_tool: bool = False):
    """以子进程方式跑一次探针，返回 (退出码, 输出)。"""
    cmd = [
        sys.executable,
        str(PROBE),
        "--db",
        str(db),
        "--thread",
        thread,
        "--counter",
        str(counter),
    ]
    if kill_after_first_tool:
        cmd.append("--kill-after-first-tool")
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=180)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _count(counter: Path) -> int:
    if not counter.exists():
        return 0
    return sum(1 for _ in counter.open("r", encoding="utf-8"))


def test_resume_after_process_restart_does_not_reexecute_tool(tmp_path):
    """进程 A 跑完一轮 → 进程 B 用同 thread 重发：不重复执行工具，且结果一致。

    这条守的是「重发/重试不产生重复副作用」——与进程内那条测试同一语义，
    但这里跨的是**进程边界**：checkpoint 必须真的落到了 SQLite 上。
    """
    db = tmp_path / "ckpt.sqlite"
    counter = tmp_path / "runs.txt"
    thread = "sess-x:t0"

    code1, out1 = _run_probe(db, thread, counter)
    assert code1 == 0, f"首次运行失败：\n{out1}"
    assert "backend=sqlite" in out1, f"首次运行未用上 sqlite 后端：\n{out1}"
    assert _count(counter) == 1, "首次运行应只执行一次工具"
    assert db.exists() and db.stat().st_size > 0, "checkpoint 未落盘"

    code2, out2 = _run_probe(db, thread, counter)
    assert code2 == 0, f"恢复运行失败：\n{out2}"
    assert _count(counter) == 1, "同一 thread 重发不得重复执行已完成轮次的工具"
    assert "工具执行次数=1" in out2, f"恢复进程看到的历史不符合预期：\n{out2}"


def test_resume_after_hard_crash_finishes_the_turn(tmp_path):
    """进程在工具执行中途被硬崩（os._exit）→ 新进程同 thread 能把该轮跑完。

    语义边界（诚实版）：这里提供的是 **at-least-once**——
    已完成的节点复用、**未完成的节点会重跑**，所以工具执行次数会从 1 变成 2。
    我们不假装 exactly-once：真正要防重复副作用的工具（写文件/发请求）应自己做幂等。
    """
    db = tmp_path / "ckpt.sqlite"
    counter = tmp_path / "runs.txt"
    thread = "sess-y:t0"

    code1, out1 = _run_probe(db, thread, counter, kill_after_first_tool=True)
    assert code1 == 9, f"探针应模拟硬崩（退出码 9），实际 {code1}：\n{out1}"
    assert _count(counter) == 1, "崩溃前工具应已执行一次"

    code2, out2 = _run_probe(db, thread, counter)
    assert code2 == 0, f"崩溃后的续跑失败：\n{out2}"
    assert _count(counter) == 2, "未完成的工具节点会重跑一次（at-least-once）"
    assert "最终答案" in out2, f"续跑应能收束出最终答案：\n{out2}"


def test_probe_helper_is_not_shipped_as_test(tmp_path):
    """自检：探针脚本本身不能被 pytest 收集（文件名以下划线开头）。"""
    assert PROBE.name.startswith("_")
    assert PROBE.exists()
    assert shutil.which(sys.executable) or Path(sys.executable).exists()
