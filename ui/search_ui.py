from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import MATCH_SIMILARITY_THRESHOLD
from services.matching_service import MatchingDatabaseError, search_found_child
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


def render_found_child_search_page() -> None:
    st.title("Found Child Search")

    uploaded_image = st.file_uploader(
        "Upload found child image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded found child image", width=320)

    search_submitted = st.button("Search Missing Child Database", use_container_width=True)
    if not search_submitted:
        return

    try:
        with st.spinner("Generating embedding and searching registered children..."):
            result = search_found_child(uploaded_image)

        st.success(f"Search completed. Search ID: {result['search_id']}")
        st.caption(
            f"Candidates checked: {result['candidate_count']} | "
            f"Best similarity: {result['best_similarity_score'] * 100:.2f}% | "
            f"Threshold: {MATCH_SIMILARITY_THRESHOLD * 100:.2f}%"
        )

        if result.get("duplicate_search_count", 0) > 0:
            st.info("This image has been searched before. Showing the latest result.")

        matches = result.get("matches", [])
        if not matches:
            st.warning("No matching child found.")
            return

        _render_matches(matches)

    except ValidationError as exc:
        st.error(str(exc))
        logger.info("Found child search validation failed: %s", exc)
    except MatchingDatabaseError as exc:
        st.error(str(exc))
        logger.exception("Found child search database failure")
    except Exception:
        st.error("Search could not be completed. Please check the logs and try again.")
        logger.exception("Unexpected found child search UI failure")


def _render_matches(matches: list[dict[str, Any]]) -> None:
    st.subheader("Top Matching Records")

    for index, match in enumerate(matches, start=1):
        title = (
            f"#{index} | {match['case_id']} | {match['child_name']} | "
            f"{match['similarity_percentage']:.2f}%"
        )
        with st.expander(title, expanded=index == 1):
            image_col, details_col = st.columns([1, 2])

            with image_col:
                _render_stored_child_image(match.get("stored_image_path"))

            with details_col:
                st.write(f"Case ID: {_display_value(match.get('case_id'))}")
                st.write(f"Child Name: {_display_value(match.get('child_name'))}")
                st.write(f"Age: {_display_value(match.get('age'))}")
                st.write(f"Gender: {_display_value(match.get('gender'))}")
                st.write(f"Guardian Name: {_display_value(match.get('guardian_name'))}")
                st.write(f"Contact Number: {_display_value(match.get('contact_number'))}")
                st.write(f"Similarity Score: {match['similarity_percentage']:.2f}%")


def _render_stored_child_image(image_path_value: str | None) -> None:
    if not image_path_value:
        st.info("Stored child image is unavailable.")
        return

    image_path = Path(image_path_value)
    if not image_path.exists() or not image_path.is_file():
        st.info("Stored child image is unavailable.")
        return

    st.image(str(image_path), caption="Stored child image", width=260)


def _display_value(value: Any) -> str:
    if value is None:
        return "Not provided"
    if isinstance(value, str) and not value.strip():
        return "Not provided"
    return str(value)
