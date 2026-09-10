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


def glass_card(title: str, body_html: str) -> None:
    """玻璃卡片。"""
    import streamlit as st

    st.markdown(
        f'<div class="glass-card"><div class="panel-title">{title}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def typing_indicator() -> str:
    return '<div class="typing"><span></span><span></span><span></span></div>'
