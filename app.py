"""GenAI Career Assistant —— Streamlit 演示入口。

布局：左侧边栏（会话/知识库/配置） + 主对话区 + 右侧产物与评测面板。
业务规则全部在 src 下，本文件只做渲染与事件处理。
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.cache import get_cost_tracker  # noqa: E402
from src.embeddings import check_embedding_health  # noqa: E402
from src.graph.workflow import route_only  # noqa: E402
from src.llm import check_llm_health  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402
from src.rag.pipeline import RetrievalConfig  # noqa: E402
from src.rag.registry import get_registry  # noqa: E402
from src.session import SessionManager  # noqa: E402
from src.state import MODE_LABELS, MULTI_TURN_MODES  # noqa: E402
from src.storage import list_outputs, read_output  # noqa: E402
from src.telemetry import setup_telemetry  # noqa: E402
from src.ui_style import brand_header, glass_card, inject_style, route_chip, warn_box  # noqa: E402

setup_telemetry()  # 仅当配置了 OTEL_EXPORTER_OTLP_ENDPOINT 时启用导出

logger = get_logger(__name__)

st.set_page_config(
    page_title="GenAI Career Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_style()

EXAMPLES = [
    "帮我写一篇 LangGraph 实战教程",
    "RAG 和微调该怎么选？",
    "我要按这份 JD 改简历",
    "整理 20 道 GenAI 面试题",
    "我想做一场模拟面试",
    "长沙有哪些 AI 应用工程师岗位",
    "根据知识库回答：切分策略怎么选？",
]

MODE_OPTIONS = ["(自动路由)"] + [
    f"{v}（{k}）" for k, v in MODE_LABELS.items() if k != "fallback"
]


# ---------------------------------------------------------------- 状态
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("manager", None)
    ss.setdefault("mode", None)
    ss.setdefault("mode_reason", "")
    ss.setdefault("kb_name", None)
    ss.setdefault("retrieval_cfg", RetrievalConfig())
    ss.setdefault("pending", None)
    ss.setdefault("manual_mode", "(自动路由)")
    ss.setdefault("last_artifact", None)
    ss.setdefault("stream", True)  # 流式输出开关，默认开启
    ss.setdefault("pending_query", None)  # 待生成的输入，由 render_chat 在容器内流式渲染


init_state()
ss = st.session_state


# ---------------------------------------------------------------- 缓存资源
@st.cache_resource(show_spinner=False)
def _registry():
    return get_registry()


@st.cache_data(show_spinner=False, ttl=120)
def _health():
    llm_ok, llm_detail = check_llm_health()
    emb_ok, emb_detail = check_embedding_health()
    return llm_ok, llm_detail, emb_ok, emb_detail


registry = _registry()


# ---------------------------------------------------------------- 侧边栏
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ⚙️ 会话与配置")

        if st.button("＋ 新建会话", use_container_width=True):
            ss.messages = []
            ss.manager = None
            ss.mode = None
            ss.last_artifact = None
            st.rerun()

        st.selectbox("手动指定模式", MODE_OPTIONS, key="manual_mode")  # 值经 key 写入 ss.manual_mode
        st.caption("默认按你的输入自动路由，可在此强制切换。")

        st.checkbox("⚡ 流式输出（逐字显示）", key="stream")
        st.caption("关闭则等整段生成完再显示，可用于对比两种体验。")

        llm_ok, llm_detail, emb_ok, emb_detail = _health()
        kv = "".join(
            f'<div class="kv"><span>{k}</span><b>{v}</b></div>' for k, v in config.describe().items()
        )
        glass_card("🔌 运行配置", kv)
        if not llm_ok:
            st.error(f"模型未连通：{llm_detail}")
        if not emb_ok:
            st.warning(f"Embedding 未就绪：{emb_detail}")

        st.markdown("### 📚 知识库")
        kb_names = registry.names()
        if kb_names:
            ss.kb_name = st.selectbox(
                "当前知识库",
                kb_names,
                index=kb_names.index(ss.kb_name) if ss.kb_name in kb_names else 0,
            )
        else:
            st.caption("暂无知识库，上传文档后自动创建。")

        new_kb = st.text_input("新建/切换到知识库名", value=ss.kb_name or "default")
        uploaded = st.file_uploader(
            "上传文档（PDF/Word/MD/TXT/HTML）",
            type=["pdf", "docx", "md", "markdown", "txt", "html", "htm"],
            accept_multiple_files=True,
        )
        if st.button("🚧 上传并建库", use_container_width=True, disabled=not uploaded):
            build_kb(new_kb, uploaded)

        if kb_names:
            stats = registry.get(ss.kb_name).stats() if registry.get(ss.kb_name) else {}
            if stats:
                glass_card(
                    "📊 索引概况",
                    "".join(
                        f'<div class="kv"><span>{k}</span><b>{v}</b></div>'
                        for k, v in {
                            "片段数": stats.get("chunks"),
                            "文档数": stats.get("sources"),
                            "向量维度": stats.get("dim"),
                            "BM25": "已构建" if stats.get("has_bm25") else "未构建",
                        }.items()
                    ),
                )

        st.markdown("### 🔎 检索参数")
        cfg: RetrievalConfig = ss.retrieval_cfg
        cfg.use_bm25 = st.checkbox("启用 BM25", value=cfg.use_bm25)
        cfg.use_vector = st.checkbox("启用向量检索", value=cfg.use_vector)
        cfg.reranker = st.selectbox(
            "重排方式", ["none", "llm", "api"], index=["none", "llm", "api"].index(cfg.reranker)
        )
        cfg.top_k = st.slider("top_k", 1, 10, cfg.top_k)
        st.caption("改参数后重新提问即生效；也可在评测面板里做 A-B 对比。")

        st.markdown("### 🧪 评测")
        if st.button("▶️ 运行评测并生成报告", use_container_width=True):
            run_eval()

        cost = get_cost_tracker().summary()
        glass_card(
            "💰 成本统计",
            "".join(f'<div class="kv"><span>{k}</span><b>{v}</b></div>' for k, v in cost.items()),
        )


def build_kb(name: str, uploaded) -> None:
    """保存上传文件并建库。"""
    upload_dir = config.DATA_DIR / "uploads" / (name or "default")
    if upload_dir.exists():
        import shutil

        shutil.rmtree(upload_dir, ignore_errors=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for f in uploaded:
        dest = upload_dir / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        paths.append(str(dest))

    with st.spinner(f"正在解析、切分并向量化 {len(paths)} 个文档…"):
        pipeline = registry.get_or_create(name or "default")
        try:
            result = pipeline.ingest(paths)
        except Exception as exc:  # noqa: BLE001
            st.error(f"建库失败：{type(exc).__name__}: {exc}")
            return
    ss.kb_name = name
    if result.chunks:
        st.success(f"建库完成：{result.documents} 个文档 / {result.chunks} 个片段 / 维度 {result.dim}")
    else:
        st.warning("未解析出有效内容，请检查文档是否为扫描件或受保护 PDF。")
    st.rerun()


def run_eval() -> None:
    """运行评测并生成报告。"""
    from src.eval import (
        build_offline_records,
        default_experiments,
        generate_from_pipeline,
        load_records,
        run_all,
        save_records,
        save_report,
    )

    kb = ss.kb_name or (registry.names()[0] if registry.names() else None)
    if not kb:
        st.warning("请先创建知识库。")
        return
    pipeline = registry.get(kb)
    if pipeline is None or not pipeline.ready:
        st.warning("当前知识库尚未建库。")
        return

    with st.spinner("正在准备评测集…"):
        records = load_records()
        if not records:
            try:
                records = generate_from_pipeline(pipeline, per_chunk=2)
            except Exception:  # noqa: BLE001
                records = []
            if not records:
                records = build_offline_records(pipeline)
            if records:
                save_records(records)

    if not records:
        st.error("评测集为空，无法评测。")
        return

    with st.spinner(f"正在跑 {len(default_experiments())} 组实验（{len(records)} 条样本）…"):
        results = run_all(pipeline, records, default_experiments(), k=5, with_hallucination=True)
        path = save_report(results, pipeline, k=5)

    ss.last_artifact = path
    st.success(f"评测完成，报告：{Path(path).name}")
    st.rerun()


# ---------------------------------------------------------------- 主区
def _esc(text: object) -> str:
    """转义后内嵌 HTML（换行转 <br>），避免用户输入破坏结构或注入脚本。"""
    return html.escape(str(text or "")).replace("\n", "<br>")


def render_chat() -> None:
    """渲染对话区。

    关键：不能用 `st.markdown('<div>')` 打开未闭合标签再把 Streamlit 元素塞进去。
    每个 st.markdown 都是独立 HTML 片段，浏览器会自动闭合未闭合标签——
    结果是容器变成空壳（chat-shell 的 min-height 撑出一大片空白）、
    消息被渲染到容器外，气泡样式也完全失效。
    这里改用 Streamlit 原生容器承载消息，气泡样式交给 CSS 作用在原生元素上。
    """
    with st.container(height=460):
        if not ss.messages:
            st.caption("还没有对话。在下方输入，或从上方选一个示例开始。")

        for msg in ss.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🧑"):
                    # 隐藏标记：供 CSS 区分角色，避免用户气泡与容器卡片样式叠加
                    st.markdown('<span data-role="user" hidden></span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="msg-user">{_esc(msg["content"])}</div>', unsafe_allow_html=True)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    # 助手回复是 Markdown，必须交给 st.markdown 渲染，不能用 div 包裹
                    st.markdown(msg["content"])
                    if msg.get("citations"):
                        cites = "<br>".join(f"• {_esc(c)}" for c in msg["citations"])
                        st.markdown(
                            f'<div class="cite-box">📎 引用来源<br>{cites}</div>',
                            unsafe_allow_html=True,
                        )

        # 待生成：在容器内部以气泡形式流式渲染，避免先出现在容器外再跳进消息列表
        if ss.pending_query:
            _render_streaming_reply(ss.pending_query)


def render_confirm_bar() -> None:
    """人机协同：草稿产出后提示人工审阅，确认后才落盘定稿。

    高风险场景（简历 / 模拟面试 / 职位清单）或命中高风险话题时产物先落草稿，
    避免 Agent 全自动生成的内容被直接外发使用。
    """
    mgr = ss.manager
    if mgr is None or not mgr.awaiting_confirmation:
        return

    reason = (
        f"命中高风险话题「{mgr.high_risk_topic}」"
        if mgr.high_risk_topic
        else "该场景产物涉及外发或决策风险"
    )
    warn_box(
        f"⚠️ <b>草稿待确认</b>：{reason}。"
        "请先审阅上方内容，确认无误后再定稿——草稿不作为正式产物外发。"
    )
    _, c2 = st.columns([0.75, 0.25])
    with c2:
        if st.button("✅ 我已审阅，定稿", use_container_width=True, type="primary"):
            path = mgr.confirm()
            ss.last_artifact = path
            st.success(f"已定稿：{Path(path).name}")
            st.rerun()


def handle_query(query: str) -> None:
    """只做输入校验与入队。

    真正的生成放到 render_chat 内执行——这样流式内容直接落在对话容器的助手气泡里，
    不会先在容器外渲染、等生成完再跳进消息列表。
    """
    from src.safety import check_input

    ok, reason = check_input(query)
    if not ok:
        st.warning(reason)
        return

    ss.messages.append({"role": "user", "content": query})
    ss.pending_query = query


def _ensure_manager(query: str) -> None:
    """首次输入：确定模式并创建会话管理器。"""
    if ss.manager is not None:
        return
    if ss.manual_mode != "(自动路由)":
        mode = ss.manual_mode.split("（")[-1].rstrip("）")
    else:
        routed = route_only(query)
        mode = routed.get("mode", "qa")
        ss.mode_reason = routed.get("message", "")
    kb = ss.kb_name if mode == "knowledge" else None
    ss.mode = mode
    ss.manager = SessionManager(mode, kb_name=kb, retrieval_cfg=ss.retrieval_cfg)


def _error_reply(exc: Exception) -> str:
    """模型不可用时的统一提示（不暴露堆栈）。"""
    return (
        f"⚠️ 模型调用失败：{type(exc).__name__}。\n"
        "请检查 `.env` 中的 OPENAI_API_KEY / EMBEDDING_API_KEY 是否有效，"
        "以及网络是否可达；配置正确后刷新页面重试。"
    )


def _render_streaming_reply(query: str) -> None:
    """在对话容器内以助手气泡渲染本轮回复（默认逐字）。

    必须在 render_chat 的容器上下文里调用，否则气泡会渲染到容器外。
    """
    try:
        _ensure_manager(query)
        stream_iter = ss.manager.step_stream(query)  # 未开始则内部自动走 start
    except Exception as exc:  # noqa: BLE001
        logger.error("会话初始化失败：%s", exc)
        ss.messages.append({"role": "assistant", "content": _error_reply(exc), "citations": []})
        ss.pending_query = None
        return

    with st.chat_message("assistant", avatar="🤖"):
        pieces: list[str] = []
        if ss.stream:
            holder = st.empty()
            try:
                for piece in stream_iter:
                    pieces.append(piece)
                    holder.markdown("".join(pieces) + "▌")
            except Exception as exc:  # noqa: BLE001
                logger.error("对话生成失败：%s", exc)
                pieces.append(_error_reply(exc))
            holder.markdown("".join(pieces))
        else:
            try:
                with st.spinner("正在思考…"):
                    pieces.append("".join(stream_iter))
            except Exception as exc:  # noqa: BLE001
                logger.error("对话生成失败：%s", exc)
                pieces.append(_error_reply(exc))
            st.markdown("".join(pieces))

        # 引用来源与本轮回复一起渲染，否则要等下一次 rerun 才会出现
        cites = list(ss.manager.citations)
        if cites:
            cites_html = "<br>".join(f"• {_esc(c)}" for c in cites)
            st.markdown(
                f'<div class="cite-box">📎 引用来源<br>{cites_html}</div>',
                unsafe_allow_html=True,
            )

    if ss.manager.artifact_path:
        ss.last_artifact = ss.manager.artifact_path
    ss.messages.append(
        {
            "role": "assistant",
            "content": "".join(pieces),
            "citations": list(ss.manager.citations),
        }
    )
    ss.pending_query = None


def render_input_area() -> None:
    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for i, (col, ex) in enumerate(zip(cols, EXAMPLES, strict=False)):
        with col:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                ss.pending = ex
    st.markdown("</div>", unsafe_allow_html=True)

    query = st.chat_input("说说你想做什么：学 GenAI / 改简历 / 准备面试 / 找工作 / 问知识库…")
    if ss.pending:
        query = ss.pending
        ss.pending = None
    if query:
        handle_query(query)
        st.rerun()


def render_right_panel() -> None:
    st.markdown("### 📦 产物与报告")
    outputs = list_outputs()
    if not outputs:
        st.caption("还没有产物。完成一次会话或评测后会出现在这里。")
        return

    names = [f'{o["icon"]} {o["label"]} · {o["mtime"]}' for o in outputs]
    default_idx = 0
    for i, o in enumerate(outputs):
        if ss.last_artifact and o["path"] == ss.last_artifact:
            default_idx = i
            break
    choice = st.selectbox("选择产物", range(len(outputs)), format_func=lambda i: names[i], index=default_idx)
    item = outputs[choice]

    with st.expander("预览", expanded=True):
        st.markdown(read_output(item["path"]))

    with open(item["path"], "rb") as f:
        st.download_button(
            "⬇️ 下载 Markdown",
            data=f,
            file_name=item["name"],
            mime="text/markdown",
            use_container_width=True,
        )


# ---------------------------------------------------------------- 组装
llm_ok, llm_detail, emb_ok, emb_detail = _health()
brand_header(
    "GenAI Career Assistant",
    "RAG + Agent 职业引擎｜学习 · 简历 · 面试 · 求职",
    ["教程生成", "答疑问答", "简历制作", "面试真题", "模拟面试", "职位搜索", "知识库问答"],
    llm_ok and emb_ok,
    "模型已连接" if (llm_ok and emb_ok) else ("模型未配置" if not llm_ok else "Embedding 未就绪"),
)

render_sidebar()

if ss.mode and ss.mode in MODE_LABELS:
    detail = ""
    if ss.mode == "knowledge" and ss.kb_name:
        detail = f"｜知识库：{ss.kb_name}"
    if ss.mode in MULTI_TURN_MODES:
        detail += "｜多轮模式，可继续追问；点「结束并导出」保存记录"
    route_chip(MODE_LABELS[ss.mode], detail)

left, right = st.columns([0.68, 0.32], gap="large")

with left:
    render_chat()
    render_confirm_bar()
    if ss.manager is not None and not ss.manager.finished and ss.mode in MULTI_TURN_MODES:
        c1, c2 = st.columns([0.75, 0.25])
        with c2:
            if st.button("✅ 结束并导出", use_container_width=True, type="primary"):
                path = ss.manager.finish()
                ss.last_artifact = path
                st.success(f"已导出：{Path(path).name}")
                st.rerun()
    render_input_area()

with right:
    render_right_panel()

if not llm_ok:
    st.info(
        "尚未检测到可用的大模型连接。请复制 `.env.example` 为 `.env` 并填入 OPENAI_API_KEY，"
        "然后重启应用。在配置完成前，路由、建库与检索仍可正常演示。"
    )
