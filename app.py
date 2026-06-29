from database.schema import initialize_database
from ui.records_ui import render_records_page
from ui.registration_ui import render_registration_page
from ui.search_ui import render_found_child_search_page
from utils.logger import get_logger

import streamlit as st


logger = get_logger(__name__)


def main() -> None:
    st.set_page_config(
        page_title="Missing Child Case Management",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        initialize_database()
    except Exception:
        logger.exception("Application startup failed while initializing the database")
        st.error("The system could not initialize the database. Please check the logs.")
        return

    selected_page = st.sidebar.radio(
        "Navigation",
        ["Register Missing Child", "View Records", "Found Child Search"],
    )

    if selected_page == "Register Missing Child":
        render_registration_page()
    elif selected_page == "View Records":
        render_records_page()
    else:
        render_found_child_search_page()


if __name__ == "__main__":
    main()
