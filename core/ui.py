"""Shared visual system: vibrant theme tokens, responsive layout and small UI helpers.

Everything is plain CSS injected once per page run, keyed off Streamlit's stable
``data-testid`` attributes and driven by CSS variables so the palette lives in one
place. Breakpoints: desktop >= 1025px, tablet 641-1024px, mobile <= 640px.
"""

from html import escape

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --c-indigo: #5b5bf0;
    --c-violet: #8b5cf6;
    --c-fuchsia: #d946ef;
    --c-orange: #fb923c;
    --c-cyan: #06b6d4;
    --c-emerald: #10b981;
    --c-ink: #14172b;
    --c-muted: #5d6480;
    --c-surface: #ffffff;
    --c-canvas: #f4f5ff;
    --c-line: rgba(91, 91, 240, 0.16);
    --grad-brand: linear-gradient(120deg, #5b5bf0 0%, #8b5cf6 45%, #d946ef 100%);
    --grad-warm: linear-gradient(120deg, #fb923c 0%, #f43f5e 100%);
    --radius: 16px;
    --shadow-soft: 0 10px 30px rgba(91, 91, 240, 0.12);
    --shadow-lift: 0 16px 38px rgba(91, 91, 240, 0.22);
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
    color: var(--c-ink);
}

.stApp {
    background:
        radial-gradient(1100px 520px at 0% -10%, rgba(139, 92, 246, 0.14), transparent 60%),
        radial-gradient(900px 480px at 100% 0%, rgba(251, 146, 60, 0.12), transparent 55%),
        var(--c-canvas);
}

[data-testid="stHeader"] { background: transparent; }

[data-testid="stAppViewContainer"] .main .block-container {
    width: 100%;
    max-width: 1280px;
    min-width: 0;
    padding: clamp(1rem, 2.5vw, 2.25rem) clamp(1rem, 3.5vw, 3rem) 6rem;
}

[data-testid="stAppViewContainer"] .main,
[data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"],
[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"],
[data-testid="stAppViewContainer"] [data-testid="column"] { min-width: 0; }

/* ---------- Hero ---------- */
.app-hero {
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    padding: clamp(1.1rem, 3vw, 2rem) clamp(1.1rem, 3.5vw, 2.25rem);
    margin: 0 0 1.25rem;
    color: #fff;
    background: var(--grad-brand);
    box-shadow: var(--shadow-lift);
}
.app-hero::after {
    content: "";
    position: absolute; inset: auto -60px -90px auto;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(251, 146, 60, 0.85), transparent 68%);
    pointer-events: none;
}
.app-hero h1 {
    margin: 0; padding: 0; color: #fff;
    font-size: clamp(1.45rem, 3.6vw, 2.2rem); font-weight: 800; letter-spacing: -0.02em; line-height: 1.15;
}
.app-hero p { margin: 0.45rem 0 0; max-width: 62ch; color: rgba(255, 255, 255, 0.92); font-size: clamp(0.9rem, 1.6vw, 1.02rem); }
.app-hero .chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.9rem; position: relative; z-index: 1; }
.app-hero .chip {
    padding: 0.28rem 0.7rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    background: rgba(255, 255, 255, 0.2); border: 1px solid rgba(255, 255, 255, 0.35); backdrop-filter: blur(4px);
}

/* ---------- Headings / text ---------- */
[data-testid="stMain"] h2, [data-testid="stMain"] h3 { color: var(--c-ink); letter-spacing: -0.015em; font-weight: 700; }
[data-testid="stCaptionContainer"] { color: var(--c-muted); }
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] { max-width: 100%; overflow-wrap: anywhere; }

/* ---------- Buttons ---------- */
[data-testid="stMain"] [data-testid="stButton"] > button,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button {
    border-radius: 12px; font-weight: 700; min-height: 2.6rem;
    border: 1px solid var(--c-line); background: var(--c-surface); color: var(--c-indigo);
    transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease;
}
[data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"],
[data-testid="stMain"] [data-testid="stDownloadButton"] > button[kind="primary"] {
    color: #fff; border: 0; background: var(--grad-brand); box-shadow: 0 8px 20px rgba(139, 92, 246, 0.35);
}
[data-testid="stMain"] [data-testid="stButton"] > button:hover,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-1px); box-shadow: var(--shadow-lift); filter: brightness(1.04);
}
[data-testid="stMain"] [data-testid="stButton"] > button:disabled { opacity: 0.5; transform: none; box-shadow: none; }

/* ---------- Tabs as pills ---------- */
[role="tablist"] {
    width: 100%; max-width: 100%; min-width: 0; gap: 0.35rem; padding: 0.3rem;
    background: var(--c-surface); border: 1px solid var(--c-line); border-radius: 14px;
    overflow-x: auto; overflow-y: hidden; flex-wrap: nowrap; scrollbar-width: thin; overscroll-behavior-x: contain;
}
[role="tab"] { flex-shrink: 0; white-space: nowrap; border-radius: 10px; font-weight: 600; padding: 0.45rem 0.9rem; }
[role="tab"][aria-selected="true"] { color: #fff; background: var(--grad-brand); }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }

/* ---------- Cards: metrics, expanders, alerts, containers ---------- */
[data-testid="stMetric"] {
    background: var(--c-surface); border: 1px solid var(--c-line); border-radius: var(--radius);
    padding: 0.9rem 1rem; box-shadow: var(--shadow-soft); max-width: 100%; min-width: 0;
}
[data-testid="stMetricLabel"] { color: var(--c-muted); font-weight: 600; }
[data-testid="stMetricValue"] { font-weight: 800; background: var(--grad-brand); -webkit-background-clip: text; background-clip: text; color: transparent; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] { min-width: 0; overflow-wrap: anywhere; }
[data-testid="stMain"] [data-testid="stExpander"] {
    background: var(--c-surface); border: 1px solid var(--c-line); border-radius: var(--radius); box-shadow: var(--shadow-soft);
}
[data-testid="stMain"] [data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stProgress"] > div > div > div > div { background: var(--grad-brand); }

[data-testid="stDataFrame"], [data-testid="stDataEditor"], [data-testid="stTable"] {
    max-width: 100%; min-width: 0; overflow-x: auto; overscroll-behavior-x: contain;
    border-radius: 14px; border: 1px solid var(--c-line);
}
[data-testid="stMarkdownContainer"] pre, [data-testid="stCodeBlock"] { max-width: 100%; overflow-x: auto; overscroll-behavior-x: contain; }
[data-testid="stMarkdownContainer"] img, [data-testid="stChatMessage"] img { max-width: 100%; height: auto; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    max-width: 100%; min-width: 0; border-radius: 18px; padding: 0.9rem 1rem; margin-bottom: 0.6rem;
    background: var(--c-surface); border: 1px solid var(--c-line); box-shadow: var(--shadow-soft);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(91, 91, 240, 0.10), rgba(217, 70, 239, 0.10));
}
[data-testid="stChatInput"] { max-width: 100%; min-width: 0; border-radius: 16px; box-shadow: var(--shadow-lift); }
[data-testid="stFileUploader"] { max-width: 100%; min-width: 0; }

/* ---------- Sidebar: deep indigo glass ---------- */
[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 10% 0%, rgba(139, 92, 246, 0.35), transparent 34%),
        radial-gradient(circle at 100% 24%, rgba(251, 146, 60, 0.22), transparent 36%),
        linear-gradient(180deg, #14172b 0%, #1b1f3d 55%, #10132a 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}
[data-testid="stSidebar"] > div:first-child { background: transparent; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1.1rem 0.85rem 1.5rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #f8fafc; letter-spacing: -0.02em; }
[data-testid="stSidebar"] h2 { margin-top: 0.4rem; font-size: 1.05rem; }
[data-testid="stSidebar"] h3 { font-size: 0.95rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #a8b0cf; line-height: 1.4; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #d5daf0; }
[data-testid="stSidebar"] hr { margin: 0.7rem 0; border: 0; border-top: 1px solid rgba(148, 163, 184, 0.22); }
[data-testid="stSidebar"] [data-testid="stExpander"] {
    overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.22); border-radius: 14px; box-shadow: none;
    background: linear-gradient(145deg, rgba(41, 46, 84, 0.78), rgba(20, 23, 43, 0.74));
}
[data-testid="stSidebar"] [data-testid="stExpander"] details,
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] { background: transparent !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary, [data-testid="stSidebar"] [data-testid="stExpander"] summary * { color: #f8fafc !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary { min-height: 2.6rem; color: #f8fafc; font-weight: 700; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: rgba(139, 92, 246, 0.16); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: #e5e9ff; font-weight: 600; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea, [data-testid="stSidebar"] [data-baseweb="select"] > div {
    color: #f8fafc; background: rgba(20, 23, 43, 0.85); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 10px;
}
[data-testid="stSidebar"] input:focus, [data-testid="stSidebar"] textarea:focus, [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    border-color: var(--c-fuchsia); box-shadow: 0 0 0 1px var(--c-fuchsia), 0 0 18px rgba(217, 70, 239, 0.25);
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    border: 1px dashed rgba(251, 146, 60, 0.7); border-radius: 14px; padding: 0.6rem;
    background: linear-gradient(145deg, rgba(251, 146, 60, 0.12), rgba(41, 46, 84, 0.7));
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover { border-color: #fdba74; box-shadow: 0 0 20px rgba(251, 146, 60, 0.18); }
[data-testid="stSidebar"] [data-testid="stButton"] > button, [data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
    min-height: 2.45rem; border: 0; border-radius: 11px; color: #fff; font-weight: 700;
    background: var(--grad-brand); box-shadow: 0 7px 18px rgba(91, 91, 240, 0.35);
    transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover { filter: brightness(1.1); transform: translateY(-1px); box-shadow: 0 10px 24px rgba(217, 70, 239, 0.35); }
[data-testid="stSidebar"] [data-testid="stButton"] > button:disabled { color: #7d86a8; background: rgba(41, 46, 84, 0.7); box-shadow: none; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] { border-color: var(--c-fuchsia); background: var(--grad-brand); }
[data-testid="stSidebar"] [data-testid="stAlert"] { border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.24); background: rgba(41, 46, 84, 0.78); }
[data-testid="stSidebar"] [data-testid="stMetric"] { background: rgba(41, 46, 84, 0.7); border-color: rgba(148, 163, 184, 0.22); box-shadow: none; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #a8b0cf; }

/* ---------- Evaluation metric cards ---------- */
.metric-card {
    border: 1px solid rgba(148, 163, 184, 0.28); border-radius: 14px; padding: 0.85rem 0.95rem 0.65rem;
    margin: 0.15rem 0 0.35rem; min-height: 7.2rem;
    background: linear-gradient(145deg, rgba(41, 46, 84, 0.95), rgba(20, 23, 43, 0.97));
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.22);
}
.metric-card.selected {
    border-color: var(--metric-accent, #38bdf8);
    box-shadow: 0 0 0 1px var(--metric-accent, #38bdf8), 0 10px 28px color-mix(in srgb, var(--metric-accent, #38bdf8) 30%, transparent);
}
.metric-card .metric-kicker { color: var(--metric-accent, #38bdf8); font-size: 0.73rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-card .metric-title { color: #f8fafc; font-size: 1rem; font-weight: 700; margin: 0.22rem 0; }
.metric-card .metric-description { color: #cbd5e1; font-size: 0.78rem; line-height: 1.35; }

/* ---------- Tablet ---------- */
@media (max-width: 1024px) {
    [data-testid="stAppViewContainer"] .main .block-container { padding-inline: clamp(1rem, 3vw, 1.75rem); }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.9rem !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex: 1 1 calc(50% - 0.9rem) !important; min-width: min(100%, 15rem) !important; }
}

/* ---------- Mobile ---------- */
@media (max-width: 640px) {
    [data-testid="stAppViewContainer"] .main .block-container { padding: 0.75rem clamp(0.7rem, 3vw, 1rem) 6rem; }
    .app-hero { border-radius: 18px; }
    .app-hero::after { opacity: 0.45; }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex: 1 1 100% !important; width: 100% !important; max-width: 100% !important; }
    [data-testid="stChatMessage"] { padding: 0.7rem; border-radius: 14px; }
    [data-testid="stMain"] [data-testid="stButton"] > button, [data-testid="stMain"] [data-testid="stDownloadButton"] > button { min-height: 2.85rem; width: 100%; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] [role="combobox"] { max-width: 100%; min-width: 0; font-size: 16px; }
    [role="tab"] { padding-inline: 0.65rem; font-size: 0.85rem; }
    [data-testid="stMetric"] { padding: 0.7rem 0.8rem; }
    h2 { font-size: clamp(1.15rem, 5.5vw, 1.4rem) !important; }
    h3 { font-size: clamp(1.02rem, 4.6vw, 1.2rem) !important; }
}

@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
</style>
"""


def apply_responsive_styles() -> None:
    """Inject the shared theme + responsive CSS (name kept for existing callers)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str = "", chips: tuple[str, ...] = ()) -> None:
    """Gradient page header. All text is HTML-escaped."""
    chip_html = "".join(f'<span class="chip">{escape(chip)}</span>' for chip in chips)
    chips_block = f'<div class="chips">{chip_html}</div>' if chip_html else ""
    subtitle_block = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="app-hero"><h1>{escape(title)}</h1>{subtitle_block}{chips_block}</div>',
        unsafe_allow_html=True,
    )
