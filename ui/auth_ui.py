import base64
from html import escape
from typing import Any

import streamlit as st

from config.constants import ROLE_AUTHORITY, ROLE_FINDER, ROLE_LABELS, ROLE_PARENT
from services.auth_service import (
    DEFAULT_USERS,
    AuthenticationError,
    AuthorizationError,
    authenticate_user,
)
from ui.theme import APP_NAME, APP_SUBTITLE, LOGO_PATH
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)

AUTH_USER_STATE_KEY = "auth_user"
LOGIN_ROLE_STATE_KEY = "login_role"


def get_authenticated_user() -> dict[str, Any] | None:
    user = st.session_state.get(AUTH_USER_STATE_KEY)
    return user if isinstance(user, dict) else None


def logout_user() -> None:
    st.session_state.pop(AUTH_USER_STATE_KEY, None)
    st.session_state.pop(LOGIN_ROLE_STATE_KEY, None)
    st.rerun()


def render_public_home_page() -> None:
    _render_hero()
    _render_feature_section()
    _render_login_section()


def _render_hero() -> None:
    logo_html = ""
    if LOGO_PATH.exists():
        logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        logo_html = f'<img src="data:image/svg+xml;base64,{logo_data}" alt="ChildShield AI logo" />'

    st.markdown(
        f"""
        <div class="app-home-hero">
            <div class="app-home-brand">
                {logo_html}
                <div>
                    <div class="app-home-title">{escape(APP_NAME)}</div>
                    <div class="app-home-subtitle">{escape(APP_SUBTITLE)}</div>
                </div>
            </div>
            <h1>AI-assisted missing child identification and recovery support</h1>
            <p>
                ChildShield AI combines secure case registration, OpenCV SFace facial matching,
                found-child reporting, role-based dashboards, and in-app alerts for faster review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card_col_1, card_col_2, card_col_3 = st.columns(3)
    with card_col_1:
        _navigation_card("For Parents", "Register and track missing-child cases with missing-day counters.")
    with card_col_2:
        _navigation_card("For Child Finders", "Submit a found-child report and trigger AI-assisted database search.")
    with card_col_3:
        _navigation_card("For Authorities", "Review all cases, reports, analytics, and match notifications.")


def _render_feature_section() -> None:
    st.subheader("Project Capabilities")
    feature_col_1, feature_col_2 = st.columns(2)
    with feature_col_1:
        _navigation_card(
            "Secure Case Management",
            "Structured missing-child registration, guardian details, local image storage, and audit logging.",
        )
        _navigation_card(
            "Role-Based Workflows",
            "Parents, child finders, and authorities each see only the tools needed for their responsibilities.",
        )
    with feature_col_2:
        _navigation_card(
            "AI Face Matching",
            "OpenCV YuNet detects faces and SFace embeddings support ranked found-child matching.",
        )
        _navigation_card(
            "In-App Alerts",
            "Potential AI matches create notifications for guardians and authorities for review.",
        )

    st.info(
        "Missing-child case details are protected behind role-based login. "
        "Sign in with the appropriate role to register, report, review, or verify cases."
    )


def _render_login_section() -> None:
    st.subheader("Role-Based Login")
    role_cols = st.columns(3)
    role_sequence = [ROLE_PARENT, ROLE_FINDER, ROLE_AUTHORITY]
    for column, role in zip(role_cols, role_sequence):
        with column:
            if st.button(f"Login as {ROLE_LABELS[role]}", use_container_width=True):
                st.session_state[LOGIN_ROLE_STATE_KEY] = role
                st.rerun()

    selected_role = st.session_state.get(LOGIN_ROLE_STATE_KEY, ROLE_PARENT)
    _render_login_form(selected_role)


def _render_login_form(role: str) -> None:
    demo_user = next(user for user in DEFAULT_USERS if user["role"] == role)
    with st.form(f"login_form_{role}"):
        st.markdown(f"#### {ROLE_LABELS[role]} Login")
        email = st.text_input("Email", value=demo_user["email"], max_chars=120)
        password = st.text_input("Password", type="password", value=demo_user["password"], max_chars=80)
        submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

    st.caption(f"Demo credentials: {demo_user['email']} / {demo_user['password']}")

    if not submitted:
        return

    try:
        with st.spinner("Signing in securely..."):
            user = authenticate_user(email, password, role)
        st.session_state[AUTH_USER_STATE_KEY] = user
        st.success(f"Signed in as {ROLE_LABELS[user['role']]}.")
        st.rerun()
    except ValidationError as exc:
        st.error(str(exc))
    except AuthorizationError as exc:
        st.warning(str(exc))
    except AuthenticationError as exc:
        st.error(str(exc))
        logger.info("Login failed for role=%s email=%s", role, email)
    except Exception:
        st.error("Login failed. Please check the logs and try again.")
        logger.exception("Unexpected login UI failure")


def _navigation_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="app-card">
            <div class="app-card-title">{escape(title)}</div>
            <p style="color:#475569; margin-bottom:0;">{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
