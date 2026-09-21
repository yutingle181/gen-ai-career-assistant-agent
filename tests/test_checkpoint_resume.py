"""可恢复性（P1-4）：checkpoint 降级、同轮复用、幂等产物、会话重建。

五层覆盖：
1. checkpointer 工厂的三级降级（sqlite → memory → 不启用），且**任何一种都不能抛异常**
   （含 DB 路径不可写、以及「import 正常但运行期才炸」的 saver 被可用性自检拦下）；
2. 同一 thread 重跑：已有完整 checkpoint 时直接复用结果，**不重复调用模型与工具**；
3. `save_file(key=...)` 幂等：同一会话反复收尾只覆盖同一个文件，不堆副本；
4. 会话级恢复：`SessionManager.restore` 与 `deps.restore_session` 能把历史接回来；
5. sqlite 后端自身的可用性：**工作线程里读写**（`check_same_thread=False`）与 **GC 后连接仍存活**
   （保活引用）——这两条对应 API 流式（工作线程）与「静默降级成内存」两个真实故障面。

全部离线：不联网、不依赖 API Key、不碰真实 app.db（读库函数打桩）。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src import config, storage
from src.graph import checkpoint, tool_loop
from src.session import SessionManager


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """不联网 + checkpoint 缓存不跨用例泄漏。"""
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", False)
    checkpoint.reset_checkpointer()
    yield
    checkpoint.reset_checkpointer()


@tool
def search_web(query: str) -> str:
    """伪联网检索工具（测试用）。"""
    return f"搜索结果：{query}"


# ------------------------------------------------------------------ 1. 降级工厂
def test_checkpointer_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CHECKPOINT", False)
    saver, backend = checkpoint.get_checkpointer()
    assert saver is None and backend == "disabled"


def test_checkpointer_degrades_without_sqlite(monkeypatch):
    """未装 langgraph-checkpoint-sqlite 时，必须降级而不是让服务起不来。

    这里把 sqlite 路径置空来模拟「可选依赖缺失」，断言拿到的是内存 saver
    （可用时）或 None（不可用时）——两种都算合法降级，都不允许抛异常。
    """
    monkeypatch.setattr(config, "ENABLE_CHECKPOINT", True)
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", "")
    saver, backend = checkpoint.get_checkpointer()

    assert backend in ("memory", "none")
    assert (saver is None) == (backend == "none")
    assert checkpoint.backend_name() == backend  # 结果被缓存，供 /health 展示


def test_checkpointer_degrades_when_db_path_unwritable(monkeypatch, tmp_path):
    """DB 路径不可写时只能降级，不能影响服务启动。

    目录不存在 → `sqlite3.connect` 抛 `unable to open database file`，这是启动期常态
    （挂载盘未就绪、权限不足、路径写错都会这样）。工厂必须吞掉它并降级到内存，
    否则一个「可选的可恢复性能力」会把整个服务拖起不来。
    """
    monkeypatch.setattr(config, "ENABLE_CHECKPOINT", True)
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", str(tmp_path / "not-exist-dir" / "ckpt.sqlite"))
    checkpoint.reset_checkpointer()

    saver, backend = checkpoint.get_checkpointer()  # 不允许抛异常

    assert backend in ("memory", "none")
    assert (saver is None) == (backend == "none")
    assert not (tmp_path / "not-exist-dir").exists()  # 也不该偷偷替用户建目录


def test_probe_saver_rejects_runtime_broken_saver():
    """`_probe_saver` 必须拦下「import 正常、运行期才炸」的 saver。

    回归背景（2026-09-21 实测）：`langgraph 0.2.76` 与 `langgraph-checkpoint 4.2.0` 混装时，
    import 完全正常，但真正读写才抛异常。只看 import 结果就会把这种 saver 挂到图上，
    表现为「恢复能力时好时坏」——比干脆不挂更难排查，所以自检要覆盖 setup 与读两条路径。
    """

    class _BrokenOnSetup:
        def setup(self):
            raise RuntimeError("Cannot operate on a closed database")

    class _BrokenOnRead:
        def setup(self):
            self.ready = True

        def get_tuple(self, config):
            raise RuntimeError("sqlite3.ProgrammingError: closed database")

    class _Healthy:
        def setup(self):
            self.ready = True

        def get_tuple(self, config):
            return None

    assert checkpoint._probe_saver(_BrokenOnSetup(), "sqlite") is False
    assert checkpoint._probe_saver(_BrokenOnRead(), "sqlite") is False
    assert checkpoint._probe_saver(_Healthy(), "memory") is True


def _sqlite_saver_or_skip(monkeypatch, tmp_path):
    """拿到真实 sqlite saver；环境未装可选依赖时跳过（这类用例不该假装通过）。"""
    monkeypatch.setattr(config, "ENABLE_CHECKPOINT", True)
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", str(tmp_path / "ckpt.sqlite"))
    checkpoint.reset_checkpointer()
    saver, backend = checkpoint.get_checkpointer()
    if backend != "sqlite":
        pytest.skip(f"当前环境未启用 sqlite checkpointer（backend={backend}）")
    return saver


def test_sqlite_saver_is_usable_from_worker_thread(monkeypatch, tmp_path):
    """sqlite saver 必须能在**工作线程**里读写。

    为什么这条是必需的：API 的 SSE 流式输出跑在 ASGI 的工作线程里，而 saver 是进程级单例、
    连接在启动时创建。若用默认连接参数（限制同线程使用），真实请求会报
    「SQLite objects created in a thread can only be used in that same thread」——
    单测（主线程）全绿、线上必炸。所以建连接时用了 `check_same_thread=False`，这条把它钉住。
    """
    import threading

    _sqlite_saver_or_skip(monkeypatch, tmp_path)
    model = _ScriptedModel()
    monkeypatch.setattr(tool_loop, "get_chat_model", lambda **kwargs: model)

    result: dict = {}

    def _run():
        try:
            result["out"] = tool_loop.run_with_tools(
                [HumanMessage(content="查")], tools=[search_web], thread_id="s1:t0"
            )
        except Exception as exc:  # noqa: BLE001
            result["err"] = exc

    thread = threading.Thread(target=_run, name="checkpoint-worker")
    thread.start()
    thread.join(timeout=60)
    assert not thread.is_alive(), "工作线程超时未返回（checkpoint 疑似卡住）"

    assert "err" not in result, f"工作线程里读写 checkpoint 失败：{result.get('err')}"
    assert result.get("out") == "最终答案"
    assert (tmp_path / "ckpt.sqlite").stat().st_size > 0  # 确实落盘了，不是内存兜底


def test_sqlite_saver_survives_garbage_collection(monkeypatch, tmp_path):
    """保活引用必须让连接熬过 GC。

    回归背景（2026-09-21 实测）：`SqliteSaver.from_conn_string()` 返回的是上下文管理器，
    一旦被垃圾回收就触发 `__exit__` 关掉连接，之后任何读写都报
    「Cannot operate on a closed database」。这里显式 `gc.collect()` 后再读一次——
    没有保活引用时这条必挂（表现是 sqlite 静默降级成内存）。
    """
    import gc

    saver = _sqlite_saver_or_skip(monkeypatch, tmp_path)
    gc.collect()

    assert checkpoint._KEEPALIVE, "必须持有保活引用（连接 / 上下文管理器）"
    # 空线程返回 None 属正常；关键是「没抛异常」——连接还活着
    assert saver.get_tuple({"configurable": {"thread_id": "ghost-thread"}}) is None


# ------------------------------------------------------------------ 2. 同轮复用
class _ScriptedModel:
    """按脚本返回消息并统计调用次数（用于断言「没有重复执行」）。"""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "c1"}],
            )
        return AIMessage(content="最终答案")


def test_same_thread_reuses_finished_checkpoint(monkeypatch):
    """同一 thread 重跑必须复用已有结果：模型与工具都不再执行一次。

    这条守住的是「重试不产生重复副作用」——将来接上写文件的工具时，
    重复执行才是真正危险的事。
    """
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setattr(config, "ENABLE_CHECKPOINT", True)
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", "")
    checkpoint.reset_checkpointer()

    saver, backend = checkpoint.get_checkpointer()
    if saver is None:  # 环境连内存 saver 都没有：本条不适用
        pytest.skip(f"当前环境无可用 checkpointer（backend={backend}）")
    assert isinstance(saver, InMemorySaver)

    model = _ScriptedModel()
    monkeypatch.setattr(tool_loop, "get_chat_model", lambda **kwargs: model)

    first = tool_loop.run_with_tools(
        [HumanMessage(content="查")], tools=[search_web], thread_id="s1:t0"
    )
    assert first == "最终答案"
    assert model.calls == 2  # 一轮工具 + 一次收束作答

    second = tool_loop.run_with_tools(
        [HumanMessage(content="查")], tools=[search_web], thread_id="s1:t0"
    )
    assert second == "最终答案"
    assert model.calls == 2  # 关键断言：没有重复调用模型

    # 换个 thread（下一轮）：必须是全新执行。脚本此时已切到「直接作答」分支，
    # 因此只多一次模型调用（若发生复用则会仍是 2，断言即失败）。
    tool_loop.run_with_tools([HumanMessage(content="查")], tools=[search_web], thread_id="s1:t1")
    assert model.calls == 3


def test_without_thread_id_no_checkpoint_is_used(monkeypatch):
    """不给 thread_id 时完全不挂 checkpointer：行为与改动前一致（零回归）。"""
    monkeypatch.setattr(config, "ENABLE_CHECKPOINT", True)
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", "")
    checkpoint.reset_checkpointer()

    model = _ScriptedModel()
    monkeypatch.setattr(tool_loop, "get_chat_model", lambda **kwargs: model)
    out = tool_loop.run_with_tools([HumanMessage(content="查")], tools=[search_web])
    assert out == "最终答案"
    assert model.calls == 2


# ------------------------------------------------------------------ 3. 幂等产物
def test_save_file_with_key_is_idempotent(tmp_path, monkeypatch):
    """同一幂等键反复保存 → 同一个文件被覆盖，而不是堆出副本。"""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

    first = storage.save_file("v1", "Resume", key="sess-abc")
    second = storage.save_file("v2", "Resume", key="sess-abc")

    assert first == second
    assert Path(first).read_text(encoding="utf-8") == "v2"
    assert len(list(tmp_path.glob("Resume_*.md"))) == 1


def test_save_file_without_key_keeps_timestamp_naming(tmp_path, monkeypatch):
    """不传 key 时保持旧的「类型_时间戳」命名契约（其他调用方不受影响）。"""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    path = storage.save_file("内容", "Q&A_Doubt_Session")
    assert Path(path).name.startswith("Q_A_Doubt_Session_")


# ------------------------------------------------------------------ 4. 会话重建
def test_restore_rebuilds_history_and_turn_index():
    """恢复历史后：转写记录可用于产物正文，轮次索引用于 checkpoint 隔离键。"""
    manager = SessionManager("qa")
    manager.session_id = "s1"
    assert manager.tool_thread_id == "s1:t0"
    assert manager.started is False

    manager.restore(
        [("user", "你好"), ("assistant", "你好呀"), ("user", "再问一句")]
    )

    assert manager.started is True
    assert len(manager.history) == 3
    assert manager.turn_index == 1  # 已完成的轮数 = 助手回复条数
    assert manager.tool_thread_id == "s1:t1"
    assert "**用户**：你好" in "".join(manager.record)


def test_session_forwards_thread_id_to_agent(monkeypatch, tmp_path):
    """会话必须把 checkpoint 的 thread 键透传给 Agent（否则恢复能力形同虚设）。"""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "ENABLE_HUMAN_CONFIRM", False)
    manager = SessionManager("qa")
    manager.session_id = "s1"
    seen: list = []

    def _respond(history, model=None, max_tokens=None, thread_id=None):
        seen.append(thread_id)
        return "回复"

    manager.agent.respond = _respond
    manager.start("你好")

    assert seen == ["s1:t0"]


def test_tool_thread_id_absent_without_session_id():
    """没有会话 id（本地 Streamlit 用法）时不给 thread 键，避免多会话共用一个 checkpoint。"""
    assert SessionManager("qa").tool_thread_id is None


def test_all_agents_follow_session_call_contract():
    """所有场景 Agent 的 respond / respond_stream 必须能接受会话层传入的关键字。

    回归背景：会话层是按关键字调用的（model / max_tokens / thread_id），
    子类覆写时只要漏一个参数，该场景就会被 `_generate` 的兜底吞掉——
    界面显示「模型调用失败」，看着像环境问题，实际是自己签名写残了
    （`MockInterviewAgent` 就曾漏掉 model / max_tokens，模拟面试因此不可用）。
    这条测试把「同一个调用契约」变成可强制的约束。
    """
    import inspect

    from src.agents import AGENT_CLASSES

    expected = {"model", "max_tokens", "thread_id"}
    for mode, cls in AGENT_CLASSES.items():
        for name in ("respond", "respond_stream"):
            params = inspect.signature(getattr(cls, name)).parameters
            missing = expected - set(params)
            assert not missing, f"{cls.__name__}.{name}（mode={mode}）缺少参数：{sorted(missing)}"
            for key in expected:
                assert params[key].default is None, f"{cls.__name__}.{name} 的 {key} 必须有默认值"


def test_restore_session_from_persistence(monkeypatch):
    """`deps.restore_session` 能把 SQLite 里的会话接回进程内会话表。"""
    from src.api import db, deps

    monkeypatch.setattr(config, "ENABLE_SESSION_RESUME", True)
    monkeypatch.setattr(
        db, "get_session", lambda sid: SimpleNamespace(mode="qa", kb_name="", finished=True, artifact="/tmp/a.md")
    )
    monkeypatch.setattr(
        db,
        "list_messages",
        lambda sid: [
            SimpleNamespace(role="user", content="问题"),
            SimpleNamespace(role="assistant", content="回答"),
        ],
    )
    deps.drop_session("s9")
    manager = deps.restore_session("s9")

    assert manager is not None
    assert manager.session_id == "s9"
    assert len(manager.history) == 2
    assert manager.finished is True and manager.artifact_path == "/tmp/a.md"
    assert deps.get_session_manager("s9") is manager  # 已放回进程内会话表
    deps.drop_session("s9")


def test_restore_session_returns_none_when_disabled_or_missing(monkeypatch):
    """开关关闭或库里没有该会话时返回 None（上层按新建处理），不抛异常。"""
    from src.api import db, deps

    monkeypatch.setattr(config, "ENABLE_SESSION_RESUME", False)
    assert deps.restore_session("nope") is None

    monkeypatch.setattr(config, "ENABLE_SESSION_RESUME", True)
    monkeypatch.setattr(db, "get_session", lambda sid: None)
    assert deps.restore_session("nope") is None

    def _boom(sid):
        raise RuntimeError("db is down")

    monkeypatch.setattr(db, "get_session", _boom)
    assert deps.restore_session("nope") is None  # 读库异常同样降级
