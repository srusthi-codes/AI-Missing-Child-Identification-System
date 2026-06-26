from datetime import datetime
from typing import Any

import streamlit as st

from database.repositories.record_repository import (
    SEARCH_FIELD_CASE_ID,
    SEARCH_FIELD_CHILD_NAME,
    SEARCH_FIELD_GUARDIAN_PHONE,
)
from services.records_service import DatabaseOperationError, SEARCH_FIELD_ALL, get_registered_children
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
    st.title("Registered Children Records")

    _initialize_records_state()

    with st.form("records_search_form"):
        search_label = st.selectbox("Search by", list(SEARCH_OPTIONS.keys()))
        search_value = st.text_input(
            "Search value",
            max_chars=120,
            disabled=SEARCH_OPTIONS[search_label] == SEARCH_FIELD_ALL,
        )

        action_col_1, action_col_2 = st.columns(2)
        with action_col_1:
            search_submitted = st.form_submit_button("Search Records", use_container_width=True)
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
        st.info("No registered child records found.")
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
            "Guardian": _display_value(record["guardian_name"]),
            "Phone": _display_value(record["guardian_phone"]),
            "Images": record["uploaded_image_count"],
            "Registered": _format_datetime(record["registration_date"]),
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
                st.write(f"Case ID: {_display_value(record['case_id'])}")
                st.write(f"Name: {_display_value(record['full_name'])}")
                st.write(f"Age: {_display_value(record['age'])}")
                st.write(f"Gender: {_display_value(record['gender'])}")
                st.write(f"Status: {_display_value(record['status'])}")
                st.write(f"Last seen date: {_display_value(record['last_seen_date'])}")
                st.write(f"Last seen time: {_display_value(record['last_seen_time'])}")
                st.write(f"Last seen location: {_display_value(record['last_seen_location'])}")
                st.write(f"Identification marks: {_display_value(record['identification_marks'])}")
                st.write(f"Description: {_display_value(record['description'])}")

            with parent_col:
                st.markdown("**Parent or Guardian Details**")
                st.write(f"Guardian name: {_display_value(record['guardian_name'])}")
                st.write(f"Relationship: {_display_value(record['relationship'])}")
                st.write(f"Phone: {_display_value(record['guardian_phone'])}")
                st.write(f"Email: {_display_value(record['guardian_email'])}")
                st.write(f"Address: {_display_value(record['guardian_address'])}")
                st.write(f"Government ID type: {_display_value(record['government_id_type'])}")
                st.write(f"Government ID last 4: {_display_value(record['government_id_last4'])}")
                st.write(f"Uploaded image count: {record['uploaded_image_count']}")
                st.write(f"Registration date: {_format_datetime(record['registration_date'])}")


def _search_label_from_field(search_field: str) -> str:
    for label, field in SEARCH_OPTIONS.items():
        if field == search_field:
            return label
    return "All Records"


def _format_datetime(value: Any) -> str:
    if not value:
        return "Not provided"

    raw_value = str(value)
    try:
        parsed = datetime.fromisoformat(raw_value)
        return parsed.strftime("%d %b %Y, %I:%M %p")
    except ValueError:
        return raw_value


def _display_value(value: Any) -> str:
    if value is None:
        return "Not provided"
    if isinstance(value, str) and not value.strip():
        return "Not provided"
    return str(value)
