from pathlib import Path
from typing import Any

import streamlit as st

from database.repositories.record_repository import (
    SEARCH_FIELD_CASE_ID,
    SEARCH_FIELD_CHILD_NAME,
    SEARCH_FIELD_GUARDIAN_PHONE,
)
from services.dashboard_service import DashboardDatabaseError, get_child_case_details
from services.records_service import DatabaseOperationError, SEARCH_FIELD_ALL, get_registered_children
from ui.theme import display_value, format_datetime, render_page_header
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)

SEARCH_OPTIONS = {
    "All Records": SEARCH_FIELD_ALL,
    "Case ID": SEARCH_FIELD_CASE_ID,
    "Child Name": SEARCH_FIELD_CHILD_NAME,
    "Guardian Phone Number": SEARCH_FIELD_GUARDIAN_PHONE,
}


def render_records_page() -> None:
    render_page_header(
        "Registered Children Records",
        "Search, review, and inspect registered missing-child case records.",
    )

    _initialize_records_state()

    with st.form("records_search_form"):
        search_col_1, search_col_2 = st.columns([0.9, 1.8])
        with search_col_1:
            search_label = st.selectbox("Search by", list(SEARCH_OPTIONS.keys()))
        with search_col_2:
            search_value = st.text_input(
                "Search value",
                max_chars=120,
                disabled=SEARCH_OPTIONS[search_label] == SEARCH_FIELD_ALL,
                placeholder="Enter case ID, child name, or guardian phone",
            )

        action_col_1, action_col_2 = st.columns([1, 1])
        with action_col_1:
            search_submitted = st.form_submit_button("Search Records", use_container_width=True, type="primary")
        with action_col_2:
            load_all_submitted = st.form_submit_button("Load All Records", use_container_width=True)

    if search_submitted:
        _load_records(SEARCH_OPTIONS[search_label], search_value)
    elif load_all_submitted:
        _load_records(SEARCH_FIELD_ALL, "")

    records = st.session_state.get("records_page_results", [])
    last_search_label = st.session_state.get("records_page_label", "All Records")

    st.caption(f"{len(records)} record(s) shown | Filter: {last_search_label}")

    if not records:
        st.info("No registered child records found for the selected filter.")
        return

    _render_summary_table(records)
    _render_record_details(records)


def _initialize_records_state() -> None:
    if "records_page_results" not in st.session_state:
        _load_records(SEARCH_FIELD_ALL, "")


def _load_records(search_field: str, search_value: str) -> None:
    try:
        with st.spinner("Loading records..."):
            records = get_registered_children(search_field, search_value)

        st.session_state.records_page_results = records
        st.session_state.records_page_label = _search_label_from_field(search_field)

    except ValidationError as exc:
        st.error(str(exc))
        logger.info("Records search validation failed: %s", exc)
    except DatabaseOperationError as exc:
        st.error(str(exc))
        logger.exception("Records database operation failed")
    except Exception:
        st.error("Records could not be loaded. Please check the logs and try again.")
        logger.exception("Unexpected records UI failure")


def _render_summary_table(records: list[dict[str, Any]]) -> None:
    summary_rows = [
        {
            "Case ID": record["case_id"],
            "Child Name": record["full_name"],
            "Age": record["age"],
            "Gender": record["gender"],
            "Guardian": display_value(record["guardian_name"]),
            "Phone": display_value(record["guardian_phone"]),
            "Images": record["uploaded_image_count"],
            "Registered": format_datetime(record["registration_date"]),
        }
        for record in records
    ]

    st.dataframe(summary_rows, use_container_width=True, hide_index=True)


def _render_record_details(records: list[dict[str, Any]]) -> None:
    st.subheader("Case Details")

    for record in records:
        title = f"{record['case_id']} - {record['full_name']}"
        with st.expander(title):
            child_col, parent_col = st.columns(2)

            with child_col:
                st.markdown("**Child Details**")
                st.write(f"Case ID: {display_value(record['case_id'])}")
                st.write(f"Name: {display_value(record['full_name'])}")
                st.write(f"Age: {display_value(record['age'])}")
                st.write(f"Gender: {display_value(record['gender'])}")
                st.write(f"Status: {display_value(record['status'])}")
                st.write(f"Last seen date: {display_value(record['last_seen_date'])}")
                st.write(f"Last seen time: {display_value(record['last_seen_time'])}")
                st.write(f"Last seen location: {display_value(record['last_seen_location'])}")
                st.write(f"Identification marks: {display_value(record['identification_marks'])}")
                st.write(f"Description: {display_value(record['description'])}")

            with parent_col:
                st.markdown("**Parent or Guardian Details**")
                st.write(f"Guardian name: {display_value(record['guardian_name'])}")
                st.write(f"Relationship: {display_value(record['relationship'])}")
                st.write(f"Phone: {display_value(record['guardian_phone'])}")
                st.write(f"Email: {display_value(record['guardian_email'])}")
                st.write(f"Address: {display_value(record['guardian_address'])}")
                st.write(f"Government ID type: {display_value(record['government_id_type'])}")
                st.write(f"Government ID last 4: {display_value(record['government_id_last4'])}")
                st.write(f"Uploaded image count: {record['uploaded_image_count']}")
                st.write(f"Registration date: {format_datetime(record['registration_date'])}")

            _render_record_images(record["child_id"])


def _render_record_images(child_id: int) -> None:
    try:
        case_details = get_child_case_details(child_id)
    except (ValidationError, DashboardDatabaseError):
        logger.exception("Could not load record image preview child_id=%s", child_id)
        return

    images = case_details.get("images", [])
    if not images:
        return

    st.markdown("**Uploaded Image Preview**")
    columns = st.columns(min(len(images), 4))
    for index, image in enumerate(images):
        image_path = Path(image["image_path"])
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        with columns[index % len(columns)]:
            if image_path.exists() and image_path.is_file():
                st.image(str(image_path), caption=image["original_filename"], use_container_width=True)


def _search_label_from_field(search_field: str) -> str:
    for label, field in SEARCH_OPTIONS.items():
        if field == search_field:
            return label
    return "All Records"
