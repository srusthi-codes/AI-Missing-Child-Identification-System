from config.constants import ROLE_AUTHORITY, ROLE_FINDER, ROLE_LABELS, ROLE_PARENT
from database.schema import initialize_database
from services.auth_service import ensure_default_users
from ui.age_progression_ui import render_age_progression_page
from ui.auth_ui import get_authenticated_user, logout_user, render_public_home_page
from ui.dashboard_ui import render_admin_dashboard_page
from ui.found_report_ui import render_found_child_report_page
from ui.records_ui import render_records_page
from ui.registration_ui import render_registration_page
from ui.role_dashboards_ui import (
    render_authority_dashboard_page,
    render_finder_dashboard_page,
    render_parent_dashboard_page,
)
from ui.search_ui import render_found_child_search_page
from ui.theme import LOGO_PATH, apply_app_theme, render_sidebar_brand
from utils.logger import get_logger

import streamlit as st


logger = get_logger(__name__)
DEFAULT_USERS_READY_KEY = "default_role_users_ready"


def main() -> None:
    st.set_page_config(
        page_title="ChildShield AI",
        page_icon=str(LOGO_PATH),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_app_theme()

    try:
        initialize_database()
        if not st.session_state.get(DEFAULT_USERS_READY_KEY):
            ensure_default_users()
            st.session_state[DEFAULT_USERS_READY_KEY] = True
    except Exception:
        logger.exception("Application startup failed while initializing the database")
        st.error("Database initialization failed. Check the application logs and try again.")
        return

    user = get_authenticated_user()
    if user is None:
        render_public_home_page()
        return

    render_sidebar_brand()
    st.sidebar.markdown(f"**Signed in:** {user['full_name']}")
    st.sidebar.caption(ROLE_LABELS[user["role"]])
    if st.sidebar.button("Logout", use_container_width=True):
        logout_user()

    navigation = _navigation_for_role(user["role"])
    selected_page = st.sidebar.radio("Navigation", list(navigation.keys()))
    navigation[selected_page](user)


def _navigation_for_role(role: str):
    if role == ROLE_PARENT:
        return {
            "Parent Dashboard": render_parent_dashboard_page,
            "Register Missing Child": lambda user: render_registration_page(registered_by_user_id=user["user_id"]),
        }

    if role == ROLE_FINDER:
        return {
            "Child Finder Dashboard": render_finder_dashboard_page,
            "Report Found Child": render_found_child_report_page,
            "Found Child Search": lambda user: render_found_child_search_page(),
        }

    if role == ROLE_AUTHORITY:
        return {
            "Authority Dashboard": render_authority_dashboard_page,
            "Register Missing Child": lambda user: render_registration_page(),
            "View Records": lambda user: render_records_page(),
            "Found Child Search": lambda user: render_found_child_search_page(),
            "AI Age Progression": lambda user: render_age_progression_page(),
            "Admin Dashboard": lambda user: render_admin_dashboard_page(),
        }

    return {"Home": lambda user: render_public_home_page()}


if __name__ == "__main__":
    main()
