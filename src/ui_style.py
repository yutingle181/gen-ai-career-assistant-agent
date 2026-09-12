"""Streamlit 视觉主题：深色科技风 + 玻璃拟态 + 紫青渐变。

集中在这里，方便统一调色与复用（app.py 只调用 inject_style()）。
"""

from __future__ import annotations

CUSTOM_CSS = """
<style>
/* ============ 全局背景与字体 ============ */
:root {
  --bg-0: #0B1020;
  --bg-1: #141A2E;
  --bg-2: #1B2440;
  --primary: #6D5EF8;
  --cyan: #22D3EE;
  --violet: #A78BFA;
  --text-0: #F5F7FF;
  --text-1: #A6B0D0;
  --text-2: #6B7699;
  --ok: #34D399;
  --warn: #FBBF24;
  --err: #F87171;
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(109,94,248,0.28), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,0.18), transparent 55%),
    linear-gradient(160deg, var(--bg-0) 0%, var(--bg-1) 45%, var(--bg-2) 100%);
  color: var(--text-0);
  font-family: "Noto Sans", "Microsoft YaHei", -apple-system, "Segoe UI", sans-serif;
}
[data-testid="stAppViewContainer"] { background-attachment: fixed; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(20,26,46,0.92), rgba(11,16,32,0.95));
  border-right: 1px solid rgba(255,255,255,0.06);
}
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1500px; }

/* ============ 品牌区 ============ */
.brand-wrap {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; padding: 14px 22px; margin-bottom: 14px;
  border-radius: 18px;
  background: rgba(255,255,255,0.045);
  border: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(14px);
  box-shadow: 0 12px 32px rgba(0,0,0,0.28);
  animation: fadeInUp .5s ease both;
}
.brand-title {
  font-size: 28px; font-weight: 700; letter-spacing: .3px;
  background: linear-gradient(120deg, #A78BFA 0%, #6D5EF8 45%, #22D3EE 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  margin: 0; line-height: 1.2;
}
.brand-sub { font-size: 13px; color: var(--text-2); margin-top: 2px; }
.brand-caps { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
.cap {
  font-size: 12px; padding: 4px 10px; border-radius: 999px;
  color: var(--text-1);
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.09);
  transition: all .18s ease;
}
.cap:hover { color: #fff; border-color: rgba(34,211,238,.55); transform: translateY(-1px); }

.status-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 6px; vertical-align: middle;
}
.status-ok { background: var(--cyan); box-shadow: 0 0 0 0 rgba(34,211,238,.6); animation: pulse 2s infinite; }
.status-warn { background: var(--warn); box-shadow: 0 0 0 0 rgba(251,191,36,.5); animation: pulse 2s infinite; }

/* ============ 路由提示条 ============ */
.route-chip {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 16px; margin-bottom: 12px; border-radius: 999px;
  background: linear-gradient(120deg, rgba(109,94,248,.20), rgba(34,211,238,.14));
  border: 1px solid rgba(167,139,250,.35);
  font-size: 13px; color: var(--text-0);
  animation: fadeInUp .35s ease both;
}
.route-chip b { color: #fff; }

/* ============ 聊天区 ============ */
/* 气泡样式必须作用在 Streamlit 原生 chat message 元素上：
   用 st.markdown 打开一个未闭合 <div> 去包裹后续元素是无效的——
   每个 st.markdown 是独立 HTML 片段，浏览器会自动闭合标签，
   结果容器变空壳（min-height 撑出大片空白）、消息渲染到容器外、样式完全失效。 */
[data-testid="stChatMessage"] {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  border-left: 3px solid var(--cyan);
  border-radius: 6px 18px 18px 18px;
  padding: 12px 16px;
  margin: 8px 0;
  backdrop-filter: blur(8px);
  animation: fadeInUp .28s ease both;
}
/* 用户消息内部已自带 .msg-user 气泡，容器不再叠加卡片样式 */
[data-testid="stChatMessage"]:has([data-role="user"]) {
  background: transparent;
  border: none;
  padding: 0;
  backdrop-filter: none;
}
.msg-user {
  background: linear-gradient(135deg, rgba(109,94,248,.92), rgba(167,139,250,.82));
  color: #fff; padding: 12px 16px; border-radius: 18px 18px 6px 18px;
  box-shadow: 0 8px 22px rgba(109,94,248,.25);
}
.cite-box {
  margin-top: 10px; padding: 10px 12px; border-radius: 12px;
  background: rgba(34,211,238,.07);
  border: 1px dashed rgba(34,211,238,.32);
  font-size: 12.5px; color: var(--text-1);
}

/* ============ 红色警示条（高风险 / 待人工确认） ============ */
.warn-box {
  margin: 10px 0 12px 0; padding: 12px 14px; border-radius: 12px;
  background: rgba(248,113,113,.10);
  border: 1px solid rgba(248,113,113,.42);
  border-left: 3px solid var(--err);
  font-size: 13px; color: var(--text-0);
  animation: fadeInUp .3s ease both;
}
.warn-box b { color: #FCA5A5; }

/* ============ 输入区 ============ */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px; }
.example-chip {
  font-size: 12.5px; padding: 6px 13px; border-radius: 999px; cursor: pointer;
  color: var(--text-1); background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  transition: all .18s ease;
}
.example-chip:hover {
  color: #fff; transform: translateY(-2px);
  border-color: rgba(109,94,248,.7);
  box-shadow: 0 8px 20px rgba(109,94,248,.22);
}
[data-testid="stChatInput"] textarea {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 16px !important;
  color: var(--text-0) !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: rgba(34,211,238,.65) !important;
  box-shadow: 0 0 0 3px rgba(34,211,238,.14) !important;
}

/* ============ 面板与卡片 ============ */
.glass-card {
  background: rgba(255,255,255,0.045);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px; padding: 14px 16px; margin-bottom: 12px;
  backdrop-filter: blur(12px);
  box-shadow: 0 10px 26px rgba(0,0,0,0.22);
  animation: fadeInUp .4s ease both;
}
.panel-title {
  font-size: 15px; font-weight: 600; color: var(--text-0);
  margin: 0 0 8px 0; display: flex; align-items: center; gap: 8px;
}
.kv { display: flex; justify-content: space-between; font-size: 12.5px; padding: 3px 0; color: var(--text-1); }
.kv b { color: var(--text-0); font-weight: 500; }

/* ============ 按钮 ============ */
.stButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: rgba(255,255,255,0.05) !important;
  color: var(--text-0) !important;
  transition: all .18s ease !important;
}
.stButton > button:hover {
  transform: translateY(-2px);
  border-color: rgba(34,211,238,.6) !important;
  box-shadow: 0 10px 24px rgba(34,211,238,.18) !important;
}
.stButton > button[data-baseweb="button"][kind="primary"] {
  background: linear-gradient(135deg, #6D5EF8, #22D3EE) !important;
  border: none !important; color: #fff !important;
}

/* ============ 动效 ============ */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(34,211,238,.55); }
  70% { box-shadow: 0 0 0 8px rgba(34,211,238,0); }
  100% { box-shadow: 0 0 0 0 rgba(34,211,238,0); }
}
.typing span {
  display: inline-block; width: 6px; height: 6px; margin-right: 4px;
  border-radius: 50%; background: var(--cyan);
  animation: bounce 1.2s infinite ease-in-out;
}
.typing span:nth-child(2) { animation-delay: .15s; }
.typing span:nth-child(3) { animation-delay: .3s; }
@keyframes bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: .5; }
  30% { transform: translateY(-5px); opacity: 1; }
}

/* ============ 新场景徽标（复用既有 cap 视觉，只做同色系强调） ============ */
.cap-new {
  color: #E9E6FF;
  background: linear-gradient(120deg, rgba(109,94,248,.34), rgba(34,211,238,.24));
  border-color: rgba(167,139,250,.55);
}

/* ============ 工具调用时间线（Function Calling 过程可见） ============ */
.tl-wrap {
  margin-top: 10px; padding: 10px 14px;
  border-radius: 12px;
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.07);
  border-left: 2px solid var(--primary);
  animation: fadeInUp .3s ease both;
}
.tl-head {
  display: flex; align-items: center; gap: 8px;
  font-size: 12.5px; color: var(--text-1);
}
.tl-head b { color: var(--text-0); font-weight: 600; }
.tl-step {
  display: flex; gap: 10px; padding: 8px 0; margin-top: 6px;
  border-top: 1px dashed rgba(255,255,255,.08);
}
.tl-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; flex: 0 0 auto; }
.tl-dot.ok { background: var(--ok); }
.tl-dot.fail { background: var(--err); }
.tl-dot.run { background: var(--cyan); animation: pulse 1.4s infinite; }
.tl-name {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12.5px; color: var(--text-0);
}
.tl-ms { font-size: 11.5px; color: var(--text-2); margin-left: 8px; }
.tl-args { font-size: 11.5px; color: var(--text-2); margin-top: 2px; word-break: break-word; }
.tl-sum { font-size: 12px; color: var(--text-1); margin-top: 3px; word-break: break-word; }
.tl-empty { font-size: 12px; color: var(--text-2); margin-top: 6px; }

/* ============ JD 匹配评分卡 ============ */
.score-card {
  padding: 14px 16px; border-radius: 16px;
  background: rgba(255,255,255,0.045);
  border: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(12px);
  box-shadow: 0 10px 26px rgba(0,0,0,0.22);
  animation: fadeInUp .35s ease both;
}
.score-top { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
.score-ring {
  width: 96px; height: 96px; border-radius: 50%; flex: 0 0 auto;
  display: flex; align-items: center; justify-content: center;
}
.score-ring > div {
  width: 76px; height: 76px; border-radius: 50%; background: #101733;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.score-num { font-size: 24px; font-weight: 700; color: var(--text-0); line-height: 1; }
.score-cap { font-size: 10.5px; color: var(--text-2); margin-top: 2px; }
.score-dims { flex: 1 1 220px; }
.dim { margin-bottom: 9px; }
.dim-top { display: flex; justify-content: space-between; font-size: 12.5px; color: var(--text-1); }
.dim-top b { color: var(--text-0); font-weight: 500; }
.bar { height: 6px; border-radius: 999px; background: rgba(255,255,255,.09); overflow: hidden; margin-top: 4px; }
.bar > i { display: block; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--primary), var(--cyan)); }
.score-cols { display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
.score-col {
  flex: 1 1 200px; padding: 10px 12px; border-radius: 12px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
}
.score-col.hit { border-left: 3px solid var(--ok); }
.score-col.gap { border-left: 3px solid var(--warn); }
.score-col h5 { margin: 0 0 6px; font-size: 12.5px; color: var(--text-0); font-weight: 600; }
.score-col p { font-size: 12.5px; color: var(--text-1); margin: 3px 0; }
.score-col .none { color: var(--text-2); }
.focus-strip {
  margin-top: 12px; padding: 10px 12px; border-radius: 12px;
  background: linear-gradient(120deg, rgba(109,94,248,.18), rgba(34,211,238,.12));
  border: 1px solid rgba(167,139,250,.32);
  font-size: 12.5px; color: var(--text-0);
}
.focus-strip b { color: #fff; }

/* 滚动条 */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.14); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
"""


def inject_style() -> None:
    """注入全局样式。"""
    import streamlit as st

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def brand_header(title: str, subtitle: str, caps: list[str], status_ok: bool, status_text: str) -> None:
    """顶部品牌区。"""
    import streamlit as st

    cap_html = "".join(f'<span class="cap">{c}</span>' for c in caps)
    dot_cls = "status-ok" if status_ok else "status-warn"
    st.markdown(
        f"""
        <div class="brand-wrap">
          <div>
            <div class="brand-title">{title}</div>
            <div class="brand-sub">{subtitle}</div>
          </div>
          <div style="text-align:right">
            <div class="brand-caps">{cap_html}</div>
            <div style="margin-top:8px;font-size:12px;color:#A6B0D0">
              <span class="status-dot {dot_cls}"></span>{status_text}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def route_chip(mode_label: str, detail: str) -> None:
    """路由提示条。"""
    import streamlit as st

    st.markdown(
        f'<div class="route-chip">🎯 已识别为 <b>{mode_label}</b>'
        f'<span style="color:#A6B0D0">{detail}</span></div>',
        unsafe_allow_html=True,
    )


def warn_box(body_html: str) -> None:
    """红色警示条：用于高风险提示（如草稿待人工确认）。"""
    import streamlit as st

    st.markdown(f'<div class="warn-box">{body_html}</div>', unsafe_allow_html=True)


def tool_timeline_html(events: list[dict]) -> str:
    """工具调用时间线：把 Function Calling 的每一步渲染成可读的一行。

    仅渲染 Agent 上报的摘要字段（工具名 / 耗时 / 参数与结果摘要），
    不渲染工具原文，与后端「事件不带全文」的隐私约定保持一致。
    """
    import html as _html

    def esc(value: object) -> str:
        return _html.escape(str(value if value is not None else ""))

    if not events:
        return (
            '<div class="tl-wrap"><div class="tl-head">🧩 <b>工具调用</b>'
            '<span>未调用工具，直接作答</span></div></div>'
        )

    total_ms = sum(int(e.get("elapsed_ms") or 0) for e in events)
    steps = []
    for idx, event in enumerate(events, start=1):
        ok = bool(event.get("ok", True))
        dot = "ok" if ok else "fail"
        mark = "✅" if ok else "⚠️"
        steps.append(
            f'<div class="tl-step"><span class="tl-dot {dot}"></span><div style="flex:1">'
            f'<div><span class="tl-name">{mark} {esc(event.get("name", "unknown"))}</span>'
            f'<span class="tl-ms">#{idx} · {int(event.get("elapsed_ms") or 0)}ms</span></div>'
            f'<div class="tl-args">入参：{esc(event.get("args_summary", ""))}</div>'
            f'<div class="tl-sum">返回：{esc(event.get("result_summary", ""))}</div>'
            "</div></div>"
        )
    head = (
        f'<div class="tl-head">🧩 <b>工具调用过程</b>'
        f'<span>共 {len(events)} 步 · 合计 {total_ms}ms</span></div>'
    )
    return f'<div class="tl-wrap">{head}{"".join(steps)}</div>'


def jd_scorecard_html(result, dimension_labels: dict[str, str]) -> str:
    """JD 匹配评分卡：总分环形 + 分项进度条 + 命中/缺口双栏 + 面试准备重点。"""
    import html as _html

    def esc(value: object) -> str:
        return _html.escape(str(value if value is not None else ""))

    total = max(0, min(100, int(getattr(result, "total_score", 0) or 0)))
    dims = getattr(result, "dimension_scores", None) or {}

    ring = (
        '<div class="score-ring" style="background:conic-gradient('
        f"#22D3EE 0% {total}%, rgba(255,255,255,.09) {total}% 100%);\">"
        f'<div><span class="score-num">{total}</span>'
        '<span class="score-cap">总分 / 100</span></div></div>'
    )

    bars = []
    for key, score in dims.items():
        label = dimension_labels.get(key, key)
        pct = max(0, min(100, int(score or 0)))
        bars.append(
            f'<div class="dim"><div class="dim-top"><span>{esc(label)}</span>'
            f"<b>{pct}</b></div>"
            f'<div class="bar"><i style="width:{pct}%"></i></div></div>'
        )

    def _col(title: str, items, css: str, empty: str) -> str:
        items = list(items or [])
        if items:
            body = "".join(f"<p>• {esc(i)}</p>" for i in items)
        else:
            body = f'<p class="none">{esc(empty)}</p>'
        return f'<div class="score-col {css}"><h5>{title}</h5>{body}</div>'

    cols = (
        '<div class="score-cols">'
        + _col("✅ 命中项", getattr(result, "matched", None), "hit", "暂无命中项")
        + _col("🟠 缺口项", getattr(result, "gaps", None), "gap", "暂无明显缺口")
        + "</div>"
    )

    focus_items = list(getattr(result, "interview_focus", None) or [])
    focus = ""
    if focus_items:
        focus = (
            '<div class="focus-strip">🎯 <b>面试准备重点</b><br>'
            + "<br>".join(f"• {esc(i)}" for i in focus_items)
            + "</div>"
        )

    summary = esc(getattr(result, "summary", "") or "")
    sub = f'<div style="font-size:12.5px;color:var(--text-1);margin-bottom:10px">{summary}</div>'

    return (
        f'<div class="score-card"><div class="score-top">{ring}'
        f'<div class="score-dims">{sub}{"".join(bars)}</div></div>{cols}{focus}</div>'
    )


def glass_card(title: str, body_html: str) -> None:
    """玻璃卡片。"""
    import streamlit as st

    st.markdown(
        f'<div class="glass-card"><div class="panel-title">{title}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def typing_indicator() -> str:
    return '<div class="typing"><span></span><span></span><span></span></div>'
