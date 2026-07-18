from datetime import date, time, timedelta

import streamlit as st

from config.constants import GENDER_OPTIONS, MAX_CHILD_AGE, MIN_CHILD_AGE, RELATIONSHIP_OPTIONS
from config.settings import MAX_IMAGES_PER_CHILD, MAX_UPLOAD_SIZE_MB
from services.registration_service import register_missing_child
from ui.theme import render_page_header
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


def render_registration_page(registered_by_user_id: int | None = None) -> None:
    render_page_header(
        "Missing Child Registration",
        "Create a complete case record with guardian information and clear child images.",
    )

    with st.form("missing_child_registration_form", clear_on_submit=False):
        st.markdown("#### Child Details")
        child_col_1, child_col_2, child_col_3 = st.columns([1.3, 0.8, 0.9])

        with child_col_1:
            full_name = st.text_input("Full name", max_chars=120)
            last_seen_location = st.text_input("Last seen location", max_chars=200)

        with child_col_2:
            age = st.number_input(
                "Age",
                min_value=MIN_CHILD_AGE,
                max_value=MAX_CHILD_AGE,
                value=10,
                step=1,
            )
            gender = st.selectbox("Gender", GENDER_OPTIONS)

        with child_col_3:
            last_seen_date = st.date_input(
                "Last seen date",
                value=date.today(),
                min_value=_date_years_ago(MAX_CHILD_AGE),
                max_value=date.today(),
                help=f"Select a date from the last {MAX_CHILD_AGE} years.",
            )
            last_seen_time = st.time_input(
                "Last seen time",
                value=time(12, 0),
                step=timedelta(minutes=1),
            )

        identification_marks = st.text_area("Identification marks", max_chars=500, height=90)
        description = st.text_area("Additional description", max_chars=500, height=90)

        st.markdown("#### Parent or Guardian Details")
        parent_col_1, parent_col_2 = st.columns(2)

        with parent_col_1:
            guardian_name = st.text_input("Guardian name", max_chars=120)
            relationship = st.selectbox("Relationship", RELATIONSHIP_OPTIONS)
            phone = st.text_input("Phone number", max_chars=20)

        with parent_col_2:
            email = st.text_input("Email address", max_chars=120)
            government_id_type = st.text_input("Government ID type", max_chars=60)
            government_id_last4 = st.text_input("Government ID last 4 digits", max_chars=4)

        address = st.text_area("Address", max_chars=1000, height=100)

        st.markdown("#### Child Images")
        uploaded_images = st.file_uploader(
            "Upload clear child images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help=f"Upload 1 to {MAX_IMAGES_PER_CHILD} images. Maximum {MAX_UPLOAD_SIZE_MB} MB per image.",
        )

        if uploaded_images:
            preview_columns = st.columns(min(len(uploaded_images), 3))
            for index, uploaded_image in enumerate(uploaded_images):
                with preview_columns[index % len(preview_columns)]:
                    st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)

        submitted = st.form_submit_button(
            "Register Missing Child",
            use_container_width=True,
            type="primary",
        )

    if not submitted:
        return

    child_data = {
        "full_name": full_name,
        "age": age,
        "gender": gender,
        "identification_marks": identification_marks,
        "last_seen_location": last_seen_location,
        "last_seen_date": last_seen_date,
        "last_seen_time": last_seen_time,
        "description": description,
    }

    parent_data = {
        "guardian_name": guardian_name,
        "relationship": relationship,
        "phone": phone,
        "email": email,
        "address": address,
        "government_id_type": government_id_type,
        "government_id_last4": government_id_last4,
    }

    try:
        with st.spinner("Saving registration, validating images, and generating embeddings..."):
            result = register_missing_child(
                child_data,
                parent_data,
                uploaded_images,
                registered_by_user_id=registered_by_user_id,
            )

        st.success(f"Registration saved successfully. Case ID: {result['case_id']}")
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Images Saved", result["image_count"])
        metric_col_2.metric("Face Embeddings Stored", result["embedding_count"])

    except ValidationError as exc:
        st.error(f"Please correct the registration details: {exc}")
        logger.info("Registration validation failed: %s", exc)
    except Exception:
        st.error("Registration could not be completed. Check the logs and try again.")
        logger.exception("Unexpected registration UI failure")


def _date_years_ago(years: int) -> date:
    today = date.today()
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(month=2, day=28, year=today.year - years)
