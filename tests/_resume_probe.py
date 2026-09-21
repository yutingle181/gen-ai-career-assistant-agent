"""跨进程恢复探针（P1-4）——被 tests/test_checkpoint_cross_process.py 以**子进程**方式调用。

为什么必须是子进程：要证明「进程挂掉后能续跑」，就必须真的让一个进程死掉。
所以这里刻意做成可被反复拉起的小脚本，而不是普通单测。

用法：
    python tests/_resume_probe.py --db <sqlite 路径> --thread <thread_id> --counter <计数文件>
                                 [--kill-after-first-tool]

行为：
    - 用 sqlite checkpointer 跑一轮工具闭环（脚本化模型，全程离线、不联网、不需要 API Key）；
    - 工具每执行一次就往 --counter 追加一行，用于统计「工具真实执行次数」；
    - --kill-after-first-tool：工具第一次执行后立刻 os._exit(9)，模拟进程硬崩
      （不给任何清理机会，等价于 kill -9）。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from src import config  # noqa: E402
from src.graph import checkpoint, tool_loop  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--thread", required=True)
    ap.add_argument("--counter", required=True)
    ap.add_argument("--kill-after-first-tool", action="store_true")
    args = ap.parse_args()

    counter = pathlib.Path(args.counter)
    kill_flag = pathlib.Path(str(counter) + ".kill")

    @tool
    def counted_tool(query: str) -> str:
        """计数工具：每次真实执行都会落一行记录（用于证明「没有重复执行」）。"""
        with counter.open("a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()} {query}\n")
        done = sum(1 for _ in counter.open("r", encoding="utf-8"))
        if args.kill_after_first_tool and done == 1 and not kill_flag.exists():
            kill_flag.write_text("killed", encoding="utf-8")
            print(f"  [probe] 工具第 1 次执行完毕，模拟进程硬崩（os._exit(9)）pid={os.getpid()}")
            sys.stdout.flush()
            os._exit(9)  # 硬崩：不执行任何清理与 atexit
        return f"计数结果：{query}"

    class _ScriptedModel:
        """固定脚本：第一轮请求调用工具，第二轮给出最终答案（离线、可复现）。"""

        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "counted_tool", "args": {"query": "x"}, "id": "c1"}],
                )
            return AIMessage(content="最终答案")

    config.ENABLE_CHECKPOINT = True
    config.CHECKPOINT_DB_PATH = args.db
    checkpoint.reset_checkpointer()

    model = _ScriptedModel()
    tool_loop.get_chat_model = lambda **kwargs: model  # type: ignore[assignment]

    out = tool_loop.run_with_tools(
        [HumanMessage(content="查")],
        tools=[counted_tool],
        thread_id=args.thread,
    )
    runs = sum(1 for _ in counter.open("r", encoding="utf-8")) if counter.exists() else 0
    backend = checkpoint.backend_name()
    print(f"  [probe] pid={os.getpid()} backend={backend} 工具执行次数={runs} 输出={out!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
