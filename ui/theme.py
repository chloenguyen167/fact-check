"""Theme & sidebar styling dùng chung cho toàn bộ app Streamlit."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
_BRAND_SVG = _ROOT / "static" / "brand.svg"

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {
    --ink: #143528;
    --muted: #5b6f66;
    --line: #d7e4dc;
    --card: #ffffff;
    --accent: #0f766e;
    --accent-2: #0d9488;
    --bg1: #f4faf7;
    --bg2: #e8f3ee;
}

html, body, [class*="css"] {
    font-family: "Be Vietnam Pro", sans-serif;
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(1200px 500px at 10% -10%, #d9f2e8 0%, transparent 55%),
        radial-gradient(900px 420px at 100% 0%, #cfe8f2 0%, transparent 50%),
        linear-gradient(180deg, var(--bg1) 0%, #f7fbf9 40%, var(--bg2) 100%);
}

#MainMenu, footer { visibility: hidden; }

/* ----- Sidebar ----- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fcfa 0%, #eef6f2 100%) !important;
    border-right: 1px solid var(--line) !important;
}

[data-testid="stSidebarHeader"] {
    padding-bottom: 0.25rem !important;
}

[data-testid="stSidebarNav"] {
    padding-top: 0.15rem !important;
}

[data-testid="stSidebarNav"] ul {
    gap: 0.12rem !important;
}

[data-testid="stSidebarNav"] li {
    margin: 0 !important;
}

/* Nhãn nhóm — nhỏ, không hét chữ hoa */
[data-testid="stSidebarNav"] > ul > li > span {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: none !important;
    color: var(--muted) !important;
    margin-top: 0.75rem !important;
    margin-bottom: 0.2rem !important;
    padding-left: 0.15rem !important;
}

/* Mục menu */
[data-testid="stSidebarNav"] a {
    border-radius: 10px !important;
    padding: 0.5rem 0.7rem !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: var(--ink) !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}

[data-testid="stSidebarNav"] a span {
    text-transform: none !important;
    letter-spacing: normal !important;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(15, 118, 110, 0.08) !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(15, 118, 110, 0.12) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}

[data-testid="stSidebarCollapseButton"] {
    color: var(--muted) !important;
}

/* ----- Trang kiểm chứng ----- */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 760px;
}

.hero { text-align: center; margin-bottom: 1.4rem; }

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(15, 118, 110, 0.1);
    color: var(--accent);
    border: 1px solid rgba(15, 118, 110, 0.18);
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.85rem;
}

.hero-title {
    font-family: "Fraunces", Georgia, serif;
    font-size: clamp(1.85rem, 4vw, 2.45rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 0 0 0.55rem 0;
}

.hero-sub {
    color: var(--muted);
    font-size: 1.02rem;
    line-height: 1.55;
    max-width: 34rem;
    margin: 0 auto;
}

.hint-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.95rem 1.1rem;
    margin: 0.9rem 0 1.1rem 0;
    box-shadow: 0 8px 24px rgba(20, 53, 40, 0.04);
}

.hint-title { font-weight: 700; font-size: 0.92rem; margin-bottom: 0.35rem; }
.hint-body { color: var(--muted); font-size: 0.9rem; line-height: 1.5; margin: 0; }

.verdict-card {
    border-radius: 18px;
    padding: 1.25rem 1.35rem;
    text-align: center;
    margin: 0.4rem 0 1rem 0;
    border: 1px solid transparent;
    box-shadow: 0 10px 28px rgba(20, 53, 40, 0.06);
}

.verdict-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.75;
    margin-bottom: 0.25rem;
}

.verdict-main {
    font-family: "Fraunces", Georgia, serif;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
}

.verdict-meta { margin-top: 0.45rem; font-size: 0.95rem; font-weight: 500; opacity: 0.88; }
.section-title { font-size: 1.05rem; font-weight: 700; margin: 1.2rem 0 0.55rem 0; }

.explain-box, .evidence-box {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    line-height: 1.65;
    color: #1f2937;
    font-size: 0.98rem;
}

.evidence-box {
    border-left: 4px solid var(--accent-2);
    background: #f8fcfa;
}

.footer-note {
    margin-top: 1.8rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.5;
}

div[data-testid="stTextArea"] textarea {
    border-radius: 14px !important;
    border: 1px solid var(--line) !important;
    background: #fff !important;
    font-size: 1.02rem !important;
    line-height: 1.55 !important;
    min-height: 120px !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.15) !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0f766e, #0d9488) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.2rem !important;
    box-shadow: 0 8px 18px rgba(15, 118, 110, 0.25) !important;
}

.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
    background: #ffffff !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: 999px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 0.8rem !important;
    box-shadow: none !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 2.4rem !important;
}

div[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    margin-bottom: 0.55rem;
}

/* ----- Trang quản trị ----- */
body.admin-page .block-container {
    max-width: 1180px;
    padding-top: 1.25rem !important;
}

.page-hero { margin-bottom: 1.25rem; }

.page-hero h1 {
    font-family: "Fraunces", Georgia, serif;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
}

.page-hero p {
    color: var(--muted);
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.5;
}
"""


def inject_theme() -> None:
    """Inject toàn bộ CSS — dùng st.html để tránh lộ raw CSS trên Streamlit 1.58+."""
    st.html(f"<style>{_CSS}</style>")


def setup_sidebar_brand() -> None:
    """Logo app ở đầu sidebar (Streamlit native)."""
    if _BRAND_SVG.exists():
        st.logo(str(_BRAND_SVG), size="large")


def mark_admin_page() -> None:
    """Đánh dấu trang quản trị để CSS mở rộng layout."""
    st.html(
        "<script>document.body.classList.add('admin-page');</script>",
        unsafe_allow_javascript=True,
    )
