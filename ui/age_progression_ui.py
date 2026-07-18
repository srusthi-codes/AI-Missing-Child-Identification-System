from pathlib import Path
from typing import Any

import streamlit as st

from config.constants import MAX_CHILD_AGE
from config.settings import AGE_PROGRESSION_MIN_IDENTITY_THRESHOLD, BASE_DIR
from services.age_progression_service import (
    DISCLAIMER,
    AgeProgressionDatabaseError,
    generate_age_progression_preview,
    get_age_progression_case_details,
    get_age_progression_children,
    save_age_progression_result,
)
from ui.theme import display_value, format_datetime, render_page_header
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)

PREVIEW_STATE_KEY = "age_progression_preview"
SELECTED_CHILD_STATE_KEY = "age_progression_selected_child_id"


def render_age_progression_page() -> None:
    render_page_header(
        "AI Age Progression",
        "Generate a clearly labelled possible future appearance estimate for long-term missing-child cases.",
    )

    st.warning(DISCLAIMER)

    try:
        child_records = _render_child_selector()
    except AgeProgressionDatabaseError as exc:
        st.error(str(exc))
        logger.exception("Age progression child selector failed")
        return
    except Exception:
        st.error("Age progression page could not be loaded. Please check the logs and try again.")
        logger.exception("Unexpected age progression UI load failure")
        return

    if not child_records:
        st.info("No registered child records are available for age progression.")
        return

    selected_child_id = st.session_state.get(SELECTED_CHILD_STATE_KEY)
    if not selected_child_id:
        return

    try:
        case_details = get_age_progression_case_details(selected_child_id)
    except ValidationError as exc:
        st.error(str(exc))
        logger.info("Age progression case selection validation failed: %s", exc)
        return
    except AgeProgressionDatabaseError as exc:
        st.error(str(exc))
        logger.exception("Age progression case details failed")
        return

    _render_child_summary(case_details)
    _render_generation_controls(case_details)
    _render_history(case_details.get("age_progression_history", []))


def _render_child_selector() -> list[dict[str, Any]]:
    search_value = st.text_input(
        "Search child records",
        placeholder="Search by case ID, child name, or status",
        max_chars=120,
    )
    child_records = get_age_progression_children(search_value)
    if not child_records:
        return []

    options = {f"{record['full_name']} | {record['case_id']} | Age {record['age']} | {record['status']}": record for record in child_records}
    selected_label = st.selectbox("Select registered child", list(options.keys()))
    selected_record = options[selected_label]
    previous_child_id = st.session_state.get(SELECTED_CHILD_STATE_KEY)
    if previous_child_id != selected_record["child_id"]:
        st.session_state[SELECTED_CHILD_STATE_KEY] = selected_record["child_id"]
        st.session_state.pop(PREVIEW_STATE_KEY, None)
    return child_records


def _render_child_summary(case_details: dict[str, Any]) -> None:
    st.subheader("Selected Child Summary")
    image_col, details_col = st.columns([1, 2])

    with image_col:
        images = case_details.get("images", [])
        if images:
            _render_image(images[0]["image_path"], "Registered child image")
        else:
            st.info("No registered images are available.")

    with details_col:
        st.write(f"Child Name: {display_value(case_details.get('full_name'))}")
        st.write(f"Case ID: {display_value(case_details.get('case_id'))}")
        st.write(f"Registered Age: {display_value(case_details.get('age'))}")
        st.write(f"Last Seen Date: {display_value(case_details.get('last_seen_date'))}")
        st.write(f"Case Status: {display_value(case_details.get('status'))}")
        st.write(f"Registration Date: {format_datetime(case_details.get('registration_date'))}")


def _render_generation_controls(case_details: dict[str, Any]) -> None:
    st.subheader("Generate Age-Progressed Estimate")
    images = case_details.get("images", [])
    if not images:
        st.warning("This child has no registered source images for age progression.")
        return

    source_age = int(case_details["age"])
    if source_age >= MAX_CHILD_AGE:
        st.warning(f"Target age cannot be selected because the registered age is already {MAX_CHILD_AGE}.")
        return

    source_image_options = {
        f"Image {index + 1}: {image['original_filename']}": image["image_id"]
        for index, image in enumerate(images)
    }
    selected_source_label = st.selectbox("Select source registered image", list(source_image_options.keys()))
    selected_source_image_id = source_image_options[selected_source_label]
    selected_source_image = next(image for image in images if image["image_id"] == selected_source_image_id)

    preview_cols = st.columns(min(len(images), 3))
    for index, image in enumerate(images):
        with preview_cols[index % len(preview_cols)]:
            caption = "Selected source" if image["image_id"] == selected_source_image_id else image["original_filename"]
            _render_image(image["image_path"], caption)

    default_target_age = min(source_age + 5, MAX_CHILD_AGE)
    target_age = st.number_input(
        "Target age",
        min_value=source_age + 1,
        max_value=MAX_CHILD_AGE,
        value=default_target_age,
        step=1,
    )

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Current Age", source_age)
    metric_col_2.metric("Target Age", int(target_age))
    metric_col_3.metric("Progression Period", f"{int(target_age) - source_age} years")

    generate_clicked = st.button("Generate Age-Progressed Estimate", type="primary", use_container_width=True)
    if generate_clicked:
        try:
            with st.spinner("Validating source face, generating estimate, and checking identity preservation..."):
                st.session_state[PREVIEW_STATE_KEY] = generate_age_progression_preview(
                    child_id=case_details["child_id"],
                    source_image_id=selected_source_image_id,
                    target_age=int(target_age),
                )
            st.success("Age-progressed estimate generated and validated.")
        except ValidationError as exc:
            st.error(str(exc))
            logger.info("Age progression generation validation failed: %s", exc)
            st.session_state.pop(PREVIEW_STATE_KEY, None)
        except AgeProgressionDatabaseError as exc:
            st.error(str(exc))
            logger.exception("Age progression generation database failure")
            st.session_state.pop(PREVIEW_STATE_KEY, None)
        except Exception:
            st.error("Age progression could not be generated. Please check the logs and try again.")
            logger.exception("Unexpected age progression generation UI failure")
            st.session_state.pop(PREVIEW_STATE_KEY, None)

    preview = st.session_state.get(PREVIEW_STATE_KEY)
    if preview and preview.get("child_id") == case_details["child_id"]:
        _render_generation_result(preview, selected_source_image)


def _render_generation_result(preview: dict[str, Any], source_image: dict[str, Any]) -> None:
    st.subheader("Age Progression Result")
    result_col_1, result_col_2 = st.columns(2)
    with result_col_1:
        _render_image(source_image["image_path"], "Original Registered Image")
    with result_col_2:
        st.image(_generated_image_bytes(preview), caption="AI-Generated Age-Progressed Estimate", use_container_width=True)

    st.info(DISCLAIMER)

    identity_score = float(preview["identity_score"])
    quality_text = preview["identity_quality"]
    st.write(f"Child Name: {display_value(preview.get('child_name'))}")
    st.write(f"Case ID: {display_value(preview.get('case_id'))}")
    st.write(f"Source Age: {preview['source_age']}")
    st.write(f"Target Age: {preview['target_age']} ({preview['target_age_label']})")
    st.write(f"Progression Period: {preview['progression_years']} years")
    st.write(f"Generation Status: Completed")
    st.write(f"Identity-Preservation Quality: {quality_text} ({identity_score * 100:.2f}%)")

    if identity_score < AGE_PROGRESSION_MIN_IDENTITY_THRESHOLD:
        st.warning("Identity preservation is low. Treat this estimate as weak and unsuitable for operational decisions.")
    elif "Moderate" in quality_text:
        st.warning("Identity preservation is moderate. Review the estimate carefully before saving.")
    else:
        st.success("Identity preservation check passed with a strong quality signal.")

    st.caption(preview.get("approach_notes", ""))

    if st.button("Save Age Progression Result", use_container_width=True):
        try:
            with st.spinner("Saving generated estimate and metadata..."):
                saved = save_age_progression_result(preview)
            st.success(f"Saved age progression result. History ID: {saved['history_id']}")
            st.session_state.pop(PREVIEW_STATE_KEY, None)
            st.rerun()
        except ValidationError as exc:
            st.error(str(exc))
            logger.info("Age progression save validation failed: %s", exc)
        except AgeProgressionDatabaseError as exc:
            st.error(str(exc))
            logger.exception("Age progression save database failure")
        except Exception:
            st.error("Age progression result could not be saved. Please check the logs and try again.")
            logger.exception("Unexpected age progression save UI failure")


def _render_history(history: list[dict[str, Any]]) -> None:
    st.subheader("Age Progression History")
    if not history:
        st.info("No saved age progression results for this child yet.")
        return

    for item in history:
        with st.expander(
            f"{item['target_age_label']} | Target age {item['target_age']} | {format_datetime(item['created_at'])}",
            expanded=False,
        ):
            image_col, detail_col = st.columns([1, 2])
            with image_col:
                _render_image(item["generated_image_path"], "Saved age-progressed estimate")
            with detail_col:
                st.write(f"Source Age: {item['source_age']}")
                st.write(f"Target Age: {item['target_age']}")
                st.write(f"Progression Period: {item['progression_years']} years")
                st.write(f"Model/Approach: {item['model_name']}")
                score = item.get("identity_score")
                score_text = "Not available" if score is None else f"{score * 100:.2f}%"
                st.write(f"Identity-Preservation Quality: {item['identity_quality']} ({score_text})")
                st.caption(DISCLAIMER)


def _render_image(path_value: str, caption: str) -> None:
    image_path = _absolute_path(path_value)
    if not image_path.exists() or not image_path.is_file():
        st.info("Image file is unavailable.")
        return
    st.image(str(image_path), caption=caption, use_container_width=True)


def _generated_image_bytes(preview: dict[str, Any]) -> bytes:
    import base64

    return base64.b64decode(preview["generated_image_b64"].encode("ascii"))


def _absolute_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path
