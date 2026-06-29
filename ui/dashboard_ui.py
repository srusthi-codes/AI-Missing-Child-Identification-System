from datetime import date, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from config.constants import GENDER_OPTIONS
from services.dashboard_service import (
    DashboardDatabaseError,
    delete_child_case,
    export_child_records_csv,
    export_search_history_csv,
    get_child_case_details,
    get_dashboard_overview,
    search_dashboard_child_records,
)
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


def render_admin_dashboard_page() -> None:
    st.title("Admin Dashboard")

    try:
        overview = get_dashboard_overview()
    except DashboardDatabaseError as exc:
        st.error(str(exc))
        logger.exception("Admin dashboard overview failed")
        return
    except Exception:
        st.error("Dashboard could not be loaded. Please check the logs and try again.")
        logger.exception("Unexpected admin dashboard failure")
        return

    _render_statistics(overview["statistics"])
    _render_recent_tables(overview["recent_registrations"], overview["recent_search_history"])
    _render_filter_and_case_tools()


def _render_statistics(statistics: dict[str, Any]) -> None:
    st.subheader("System Statistics")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Registered Children", statistics["total_registered_children"])
    metric_cols[1].metric("Face Embeddings", statistics["total_face_embeddings"])
    metric_cols[2].metric("Search Requests", statistics["total_search_requests"])

    metric_cols = st.columns(3)
    metric_cols[0].metric("Successful Matches", statistics["total_successful_matches"])
    metric_cols[1].metric("Unsuccessful Searches", statistics["total_unsuccessful_searches"])
    metric_cols[2].metric("Match Success", f"{statistics['match_success_percentage']:.2f}%")


def _render_recent_tables(
    recent_registrations: list[dict[str, Any]],
    recent_search_history: list[dict[str, Any]],
) -> None:
    st.subheader("Recent Missing Child Registrations")
    if recent_registrations:
        st.dataframe(_child_table_rows(recent_registrations), use_container_width=True, hide_index=True)
    else:
        st.info("No missing child registrations found.")

    st.subheader("Recent Found-Child Search History")
    if recent_search_history:
        st.dataframe(_search_history_rows(recent_search_history), use_container_width=True, hide_index=True)
    else:
        st.info("No found-child searches found.")


def _render_filter_and_case_tools() -> None:
    st.subheader("Child Records")

    if "dashboard_child_records" not in st.session_state:
        _load_dashboard_child_records({})

    with st.form("dashboard_child_filters"):
        filter_col_1, filter_col_2, filter_col_3 = st.columns(3)
        with filter_col_1:
            case_id = st.text_input("Case ID", max_chars=80)
            gender = st.selectbox("Gender", ["All", *GENDER_OPTIONS])
        with filter_col_2:
            child_name = st.text_input("Child Name", max_chars=120)
            age = st.text_input("Age", max_chars=2)
        with filter_col_3:
            use_registration_date = st.checkbox("Filter by Registration Date")
            registration_date = st.date_input(
                "Registration Date",
                value=date.today(),
                max_value=date.today(),
                disabled=not use_registration_date,
            )

        filter_submitted = st.form_submit_button("Apply Filters", use_container_width=True)

    filters = {
        "case_id": case_id,
        "child_name": child_name,
        "gender": None if gender == "All" else gender,
        "age": age,
        "registration_date": registration_date if use_registration_date else None,
    }

    if filter_submitted:
        _load_dashboard_child_records(filters)
        st.session_state.dashboard_child_filters = filters

    active_filters = st.session_state.get("dashboard_child_filters", {})
    records = st.session_state.get("dashboard_child_records", [])

    export_col_1, export_col_2 = st.columns(2)
    with export_col_1:
        _render_child_export_button(active_filters)
    with export_col_2:
        _render_search_history_export_button()

    st.caption(f"{len(records)} child record(s) shown")
    if records:
        st.dataframe(_child_table_rows(records), use_container_width=True, hide_index=True)
        _render_case_details_selector(records)
    else:
        st.info("No child records match the selected filters.")


def _load_dashboard_child_records(filters: dict[str, Any]) -> None:
    try:
        with st.spinner("Loading child records..."):
            st.session_state.dashboard_child_records = search_dashboard_child_records(filters)
            st.session_state.dashboard_child_filters = filters
    except ValidationError as exc:
        st.error(str(exc))
        logger.info("Dashboard filter validation failed: %s", exc)
    except DashboardDatabaseError as exc:
        st.error(str(exc))
        logger.exception("Dashboard child record loading failed")
    except Exception:
        st.error("Child records could not be loaded. Please check the logs and try again.")
        logger.exception("Unexpected dashboard child record failure")


def _render_child_export_button(filters: dict[str, Any]) -> None:
    try:
        csv_data = export_child_records_csv(filters)
    except Exception:
        csv_data = b""
        logger.exception("Could not prepare child records CSV export")

    st.download_button(
        "Export Child Records CSV",
        data=csv_data,
        file_name="child_records_export.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not csv_data,
    )


def _render_search_history_export_button() -> None:
    try:
        csv_data = export_search_history_csv()
    except Exception:
        csv_data = b""
        logger.exception("Could not prepare search history CSV export")

    st.download_button(
        "Export Search History CSV",
        data=csv_data,
        file_name="search_history_export.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not csv_data,
    )


def _render_case_details_selector(records: list[dict[str, Any]]) -> None:
    st.subheader("Case Details")
    options = {f"{record['case_id']} - {record['full_name']}": record["child_id"] for record in records}
    selected_label = st.selectbox("Select case", list(options.keys()))

    if st.button("Load Case Details", use_container_width=True):
        st.session_state.dashboard_selected_child_id = options[selected_label]

    selected_child_id = st.session_state.get("dashboard_selected_child_id")
    if not selected_child_id:
        return

    try:
        case_details = get_child_case_details(selected_child_id)
    except ValidationError as exc:
        st.error(str(exc))
        logger.info("Dashboard case details validation failed: %s", exc)
        return
    except DashboardDatabaseError as exc:
        st.error(str(exc))
        logger.exception("Dashboard case details loading failed")
        return

    _render_case_details(case_details)


def _render_case_details(case_details: dict[str, Any]) -> None:
    child_col, parent_col = st.columns(2)

    with child_col:
        st.markdown("**Child Details**")
        st.write(f"Case ID: {_display_value(case_details['case_id'])}")
        st.write(f"Name: {_display_value(case_details['full_name'])}")
        st.write(f"Age: {_display_value(case_details['age'])}")
        st.write(f"Gender: {_display_value(case_details['gender'])}")
        st.write(f"Status: {_display_value(case_details['status'])}")
        st.write(f"Last seen date: {_display_value(case_details['last_seen_date'])}")
        st.write(f"Last seen time: {_display_value(case_details['last_seen_time'])}")
        st.write(f"Last seen location: {_display_value(case_details['last_seen_location'])}")
        st.write(f"Identification marks: {_display_value(case_details['identification_marks'])}")
        st.write(f"Description: {_display_value(case_details['description'])}")
        st.write(f"Registration date: {_format_datetime(case_details['registration_date'])}")

    with parent_col:
        st.markdown("**Parent or Guardian Details**")
        st.write(f"Guardian name: {_display_value(case_details['guardian_name'])}")
        st.write(f"Relationship: {_display_value(case_details['relationship'])}")
        st.write(f"Phone: {_display_value(case_details['guardian_phone'])}")
        st.write(f"Email: {_display_value(case_details['guardian_email'])}")
        st.write(f"Address: {_display_value(case_details['guardian_address'])}")
        st.write(f"Government ID type: {_display_value(case_details['government_id_type'])}")
        st.write(f"Government ID last 4: {_display_value(case_details['government_id_last4'])}")
        st.write(f"Uploaded image count: {case_details['uploaded_image_count']}")
        st.write(f"Embedding status: {case_details['embedding_status']}")
        st.write(f"Stored embeddings: {case_details['embedding_count']}")

    _render_uploaded_images(case_details["images"])
    _render_embedding_status(case_details["embeddings"])

    if st.button("Delete Selected Case", type="primary"):
        _confirm_delete_dialog(case_details)


def _render_uploaded_images(images: list[dict[str, Any]]) -> None:
    st.markdown("**Uploaded Images**")
    if not images:
        st.info("No uploaded images found for this case.")
        return

    columns = st.columns(min(len(images), 3))
    for index, image in enumerate(images):
        image_path = Path(image["image_path"])
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        with columns[index % len(columns)]:
            if image_path.exists() and image_path.is_file():
                st.image(str(image_path), caption=image["original_filename"], width=220)
            else:
                st.info("Image file is unavailable.")


def _render_embedding_status(embeddings: list[dict[str, Any]]) -> None:
    st.markdown("**Embedding Status**")
    if not embeddings:
        st.warning("No embeddings stored for this case.")
        return

    rows = [
        {
            "Embedding ID": embedding["embedding_id"],
            "Model": embedding["model_name"],
            "Detector": embedding["detector_backend"],
            "Dimension": embedding["embedding_dimension"],
            "Quality": embedding["quality_score"],
            "Face Confidence": embedding["face_confidence"],
            "Created": _format_datetime(embedding["created_at"]),
        }
        for embedding in embeddings
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


@st.dialog("Confirm Case Deletion")
def _confirm_delete_dialog(case_details: dict[str, Any]) -> None:
    st.warning(f"This will permanently delete case {case_details['case_id']} from the database.")
    confirmation = st.text_input("Type the Case ID to confirm deletion")

    if st.button("Confirm Delete", type="primary", use_container_width=True):
        try:
            result = delete_child_case(case_details["child_id"], confirmation)
            st.success(f"Deleted case {result['case_id']}.")
            st.session_state.pop("dashboard_selected_child_id", None)
            st.session_state.pop("dashboard_child_records", None)
            st.rerun()
        except ValidationError as exc:
            st.error(str(exc))
            logger.info("Dashboard delete validation failed: %s", exc)
        except DashboardDatabaseError as exc:
            st.error(str(exc))
            logger.exception("Dashboard delete failed")
        except Exception:
            st.error("Case could not be deleted. Please check the logs and try again.")
            logger.exception("Unexpected dashboard delete failure")


def _child_table_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Case ID": record["case_id"],
            "Child Name": record["full_name"],
            "Age": record["age"],
            "Gender": record["gender"],
            "Status": record["status"],
            "Guardian": _display_value(record.get("guardian_name")),
            "Phone": _display_value(record.get("guardian_phone")),
            "Images": record["uploaded_image_count"],
            "Embeddings": record["embedding_count"],
            "Registered": _format_datetime(record["registration_date"]),
        }
        for record in records
    ]


def _search_history_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Search ID": record["search_id"],
            "Matches Found": record["matches_found"],
            "Best Similarity": f"{record['best_similarity_score'] * 100:.2f}%",
            "Status": record["status"],
            "Created": _format_datetime(record["created_at"]),
        }
        for record in records
    ]


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
