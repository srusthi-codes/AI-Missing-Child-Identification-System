from pathlib import Path
from typing import Any

import streamlit as st

from services.role_dashboard_service import RoleDashboardError, submit_found_child_report
from ui.theme import display_value, render_page_header
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


def render_found_child_report_page(user: dict[str, Any]) -> None:
    render_page_header(
        "Report Found Child",
        "Submit a found-child report and run AI-assisted matching against registered cases.",
    )

    with st.form("found_child_report_form", clear_on_submit=False):
        uploaded_image = st.file_uploader(
            "Upload found child image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Use one clear image with one visible face.",
        )
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Found child image preview", width=260)

        location = st.text_input("Found location", max_chars=200, placeholder="Area, landmark, city")
        description = st.text_area(
            "Description",
            max_chars=500,
            height=120,
            placeholder="Clothing, condition, nearby details, and any urgent context",
        )
        submitted = st.form_submit_button("Submit Report and Run AI Match", type="primary", use_container_width=True)

    if not submitted:
        return

    try:
        with st.spinner("Saving report, generating SFace embedding, searching cases, and sending alerts..."):
            result = submit_found_child_report(user, uploaded_image, location, description)

        st.success(f"Found-child report submitted. Report ID: {result['report_id']}")
        st.caption(
            f"Search ID: {result['search_id']} | Matches found: {result['matches_found']} | "
            f"Best similarity: {result['best_similarity_score'] * 100:.2f}% | "
            f"Notifications created: {result['notification_count']}"
        )

        if not result["matches"]:
            st.warning("No matching child found above the configured threshold.")
            return

        st.subheader("AI Match Results")
        _render_report_matches(result["matches"])
    except ValidationError as exc:
        st.error(f"Please correct the found-child report details: {exc}")
        logger.info("Found report validation failed: %s", exc)
    except RoleDashboardError as exc:
        st.error(str(exc))
        logger.exception("Found report service failed")
    except Exception:
        st.error("Found-child report could not be submitted. Please check the logs and try again.")
        logger.exception("Unexpected found report UI failure")


def _render_report_matches(matches: list[dict[str, Any]]) -> None:
    for index, match in enumerate(matches, start=1):
        st.markdown(
            f"""
            <div class="app-match-card {'app-best-match' if index == 1 else ''}">
                <span class="app-pill">{'Best Match' if index == 1 else f'Rank {index}'}</span>
                <strong style="margin-left:.45rem;">{match['case_id']} - {match['child_name']}</strong>
                <div style="margin-top:.45rem;color:#334155;">
                    Similarity: <strong>{match['similarity_percentage']:.2f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        image_path = match.get("stored_image_path")
        with st.expander(f"Review match details for {match['case_id']}", expanded=index == 1):
            image_col, details_col = st.columns([1, 2])
            with image_col:
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption="Registered child image", use_container_width=True)
                else:
                    st.info("Stored child image unavailable.")
            with details_col:
                st.write(f"Case ID: {display_value(match.get('case_id'))}")
                st.write(f"Child Name: {display_value(match.get('child_name'))}")
                st.write(f"Age: {display_value(match.get('age'))}")
                st.write(f"Gender: {display_value(match.get('gender'))}")
                st.write(f"Guardian: {display_value(match.get('guardian_name'))}")
                st.write(f"Contact: {display_value(match.get('contact_number'))}")
                st.write(f"Similarity: {match['similarity_percentage']:.2f}%")
