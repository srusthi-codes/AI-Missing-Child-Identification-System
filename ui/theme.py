from html import escape
from typing import Any

import streamlit as st

from config.settings import BASE_DIR


APP_NAME = "ChildShield AI"
APP_SUBTITLE = "Missing Child Identification"
LOGO_PATH = BASE_DIR / "assets" / "app_logo.svg"


def apply_app_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-primary: #0f766e;
            --app-primary-dark: #115e59;
            --app-accent: #0284c7;
            --app-bg: #f6fafb;
            --app-surface: #ffffff;
            --app-muted: #64748b;
            --app-border: #dbe8ee;
            --app-danger: #b91c1c;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fcfd 0%, #eef8fa 100%);
            border-right: 1px solid var(--app-border);
        }

        [data-testid="stSidebar"] img {
            margin-top: 0.25rem;
        }

        .app-brand-title {
            color: #0f172a;
            font-size: 1.08rem;
            font-weight: 750;
            line-height: 1.15;
            margin: 0.25rem 0 0;
        }

        .app-brand-subtitle {
            color: var(--app-muted);
            font-size: 0.78rem;
            margin-bottom: 1.1rem;
        }

        .app-page-header {
            border: 1px solid var(--app-border);
            background: linear-gradient(135deg, #ffffff 0%, #f2fbfb 100%);
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
        }

        .app-page-header h1 {
            color: #0f172a;
            font-size: 1.9rem;
            line-height: 1.2;
            margin: 0;
            letter-spacing: 0;
        }

        .app-page-header p {
            color: var(--app-muted);
            font-size: 0.95rem;
            margin: 0.3rem 0 0;
        }

        .app-card {
            border: 1px solid var(--app-border);
            background: var(--app-surface);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.9rem;
        }

        .app-card-title {
            color: #0f172a;
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.6rem;
        }

        .app-home-hero {
            border: 1px solid var(--app-border);
            background: linear-gradient(135deg, #ffffff 0%, #ecfeff 58%, #eff6ff 100%);
            border-radius: 8px;
            padding: 1.3rem 1.4rem;
            margin-bottom: 1rem;
        }

        .app-home-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin-bottom: 0.8rem;
        }

        .app-home-brand img {
            width: 58px;
            height: 58px;
        }

        .app-home-title {
            color: #0f172a;
            font-size: 1.15rem;
            font-weight: 800;
        }

        .app-home-subtitle {
            color: var(--app-muted);
            font-size: 0.85rem;
        }

        .app-home-hero h1 {
            color: #0f172a;
            font-size: 2.1rem;
            line-height: 1.15;
            letter-spacing: 0;
            margin: 0 0 0.55rem;
        }

        .app-home-hero p {
            color: #475569;
            font-size: 1rem;
            max-width: 860px;
            margin: 0;
        }

        .app-kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.85rem;
            margin: 0.6rem 0 1rem;
        }

        .app-kpi-card {
            border: 1px solid var(--app-border);
            background: #ffffff;
            border-radius: 8px;
            padding: 0.95rem 1rem;
        }

        .app-kpi-label {
            color: var(--app-muted);
            font-size: 0.78rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .app-kpi-value {
            color: #0f172a;
            font-size: 1.55rem;
            font-weight: 780;
            margin-top: 0.15rem;
        }

        .app-match-card {
            border: 1px solid var(--app-border);
            border-left: 5px solid var(--app-primary);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
            background: #ffffff;
        }

        .app-best-match {
            border-left-color: var(--app-accent);
            background: #f0f9ff;
        }

        .app-pill {
            display: inline-block;
            border: 1px solid #b7e4df;
            background: #ecfdf5;
            color: var(--app-primary-dark);
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .app-status-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-size: 0.76rem;
            font-weight: 750;
            border: 1px solid var(--app-border);
            color: #334155;
            background: #f8fafc;
        }

        .app-status-match {
            border-color: #86efac;
            color: #166534;
            background: #f0fdf4;
        }

        .app-status-no-match {
            border-color: #fecaca;
            color: #991b1b;
            background: #fef2f2;
        }

        .app-status-pending {
            border-color: #bae6fd;
            color: #075985;
            background: #f0f9ff;
        }

        .app-report-card {
            border: 1px solid var(--app-border);
            background: #ffffff;
            border-radius: 8px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.8rem;
        }

        .app-report-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            flex-wrap: wrap;
            margin-bottom: 0.45rem;
        }

        .app-report-meta {
            color: var(--app-muted);
            font-size: 0.86rem;
        }

        .app-danger-note {
            border: 1px solid #fecaca;
            background: #fff1f2;
            color: #7f1d1d;
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stForm"],
        div[data-testid="stFileUploader"] section {
            border-radius: 8px;
            border-color: var(--app-border);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            overflow: hidden;
        }

        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 8px;
            border-color: var(--app-primary);
            font-weight: 700;
        }

        .stButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: var(--app-primary);
            border-color: var(--app-primary);
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        @media (max-width: 760px) {
            .app-page-header {
                padding: 0.9rem;
            }
            .app-page-header h1 {
                font-size: 1.45rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=72)
    st.sidebar.markdown(
        f"""
        <div class="app-brand-title">{APP_NAME}</div>
        <div class="app-brand-subtitle">{APP_SUBTITLE}</div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="app-page-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(items: list[tuple[str, Any]]) -> None:
    cards = "".join(
        '<div class="app-kpi-card">'
        f'<div class="app-kpi-label">{escape(str(label))}</div>'
        f'<div class="app-kpi-value">{escape(str(value))}</div>'
        "</div>"
        for label, value in items
    )
    st.markdown(f'<div class="app-kpi-grid">{cards}</div>', unsafe_allow_html=True)


def display_value(value: Any) -> str:
    if value is None:
        return "Not provided"
    if isinstance(value, str) and not value.strip():
        return "Not provided"
    return str(value)


def format_datetime(value: Any) -> str:
    if not value:
        return "Not provided"

    from datetime import datetime

    raw_value = str(value)
    try:
        parsed = datetime.fromisoformat(raw_value)
        return parsed.strftime("%d %b %Y, %I:%M %p")
    except ValueError:
        return raw_value
