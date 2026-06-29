from database.schema import initialize_database
from ui.dashboard_ui import render_admin_dashboard_page
from ui.records_ui import render_records_page
from ui.registration_ui import render_registration_page
from ui.search_ui import render_found_child_search_page
from ui.theme import LOGO_PATH, apply_app_theme, render_sidebar_brand
from utils.logger import get_logger

import streamlit as st


logger = get_logger(__name__)


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
    except Exception:
        logger.exception("Application startup failed while initializing the database")
        st.error("Database initialization failed. Check the application logs and try again.")
        return

    render_sidebar_brand()

    selected_page = st.sidebar.radio(
        "Navigation",
        ["Register Missing Child", "View Records", "Found Child Search", "Admin Dashboard"],
    )

    if selected_page == "Register Missing Child":
        render_registration_page()
    elif selected_page == "View Records":
        render_records_page()
    elif selected_page == "Found Child Search":
        render_found_child_search_page()
    else:
        render_admin_dashboard_page()


if __name__ == "__main__":
    main()
