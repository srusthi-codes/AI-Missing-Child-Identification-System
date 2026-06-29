from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import MATCH_SIMILARITY_THRESHOLD
from services.matching_service import MatchingDatabaseError, search_found_child
from ui.theme import display_value, render_page_header
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


def render_found_child_search_page() -> None:
    render_page_header(
        "Found Child Search",
        "Upload a found child image and compare it against stored OpenCV SFace embeddings.",
    )

    uploaded_image = st.file_uploader(
        "Upload one clear found child image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Use a clear frontal image with one visible face for best matching accuracy.",
    )

    if uploaded_image is not None:
        preview_col, guidance_col = st.columns([1, 2])
        with preview_col:
            st.image(uploaded_image, caption="Uploaded found child image", use_container_width=True)
        with guidance_col:
            st.info(
                "The system validates the image, generates an OpenCV SFace embedding, "
                "and compares it with registered missing-child embeddings."
            )

    search_submitted = st.button("Search Missing Child Database", use_container_width=True, type="primary")
    if not search_submitted:
        return

    try:
        with st.spinner("Validating image, generating embedding, and ranking Top-5 matches..."):
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
        st.error(f"Search validation failed: {exc}")
        logger.info("Found child search validation failed: %s", exc)
    except MatchingDatabaseError as exc:
        st.error(str(exc))
        logger.exception("Found child search database failure")
    except Exception:
        st.error("Search could not be completed. Check the logs and try again.")
        logger.exception("Unexpected found child search UI failure")


def _render_matches(matches: list[dict[str, Any]]) -> None:
    st.subheader("Top-5 Matching Records")

    for index, match in enumerate(matches, start=1):
        best_label = "Best Match" if index == 1 else f"Rank {index}"
        card_class = "app-match-card app-best-match" if index == 1 else "app-match-card"
        st.markdown(
            f"""
            <div class="{card_class}">
                <span class="app-pill">{best_label}</span>
                <strong style="margin-left: .45rem;">{match['case_id']} - {match['child_name']}</strong>
                <div style="margin-top: .45rem; color: #334155;">
                    Similarity: <strong>{match['similarity_percentage']:.2f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(
            min(float(match["similarity_score"]), 1.0),
            text=f"{match['similarity_percentage']:.2f}% match",
        )

        with st.expander(f"View details for {match['case_id']}", expanded=index == 1):
            image_col, details_col = st.columns([1, 2])

            with image_col:
                _render_stored_child_image(match.get("stored_image_path"))

            with details_col:
                st.write(f"Case ID: {display_value(match.get('case_id'))}")
                st.write(f"Child Name: {display_value(match.get('child_name'))}")
                st.write(f"Age: {display_value(match.get('age'))}")
                st.write(f"Gender: {display_value(match.get('gender'))}")
                st.write(f"Guardian Name: {display_value(match.get('guardian_name'))}")
                st.write(f"Contact Number: {display_value(match.get('contact_number'))}")
                st.write(f"Similarity Score: {match['similarity_percentage']:.2f}%")


def _render_stored_child_image(image_path_value: str | None) -> None:
    if not image_path_value:
        st.info("Stored child image is unavailable.")
        return

    image_path = Path(image_path_value)
    if not image_path.exists() or not image_path.is_file():
        st.info("Stored child image is unavailable.")
        return

    st.image(str(image_path), caption="Stored child image", use_container_width=True)
