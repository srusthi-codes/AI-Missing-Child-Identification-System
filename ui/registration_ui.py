from datetime import date

import streamlit as st

from config.constants import GENDER_OPTIONS, RELATIONSHIP_OPTIONS
from config.settings import MAX_IMAGES_PER_CHILD, MAX_UPLOAD_SIZE_MB
from services.registration_service import register_missing_child
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


def render_registration_page() -> None:
    st.title("Missing Child Registration")

    with st.form("missing_child_registration_form", clear_on_submit=False):
        st.subheader("Child Details")
        child_col_1, child_col_2 = st.columns(2)

        with child_col_1:
            full_name = st.text_input("Full name", max_chars=120)
            age = st.number_input("Age", min_value=0, max_value=18, value=10, step=1)
            gender = st.selectbox("Gender", GENDER_OPTIONS)

        with child_col_2:
            last_seen_date = st.date_input("Last seen date", value=date.today(), max_value=date.today())
            last_seen_time = st.time_input("Last seen time", value=None)
            last_seen_location = st.text_input("Last seen location", max_chars=200)

        identification_marks = st.text_area("Identification marks", max_chars=500, height=90)
        description = st.text_area("Additional description", max_chars=500, height=90)

        st.subheader("Parent or Guardian Details")
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

        st.subheader("Child Images")
        uploaded_images = st.file_uploader(
            "Upload clear child images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help=f"Upload 1 to {MAX_IMAGES_PER_CHILD} images. Maximum {MAX_UPLOAD_SIZE_MB} MB per image.",
        )

        submitted = st.form_submit_button("Register Missing Child", use_container_width=True)

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
        with st.spinner("Saving registration..."):
            result = register_missing_child(child_data, parent_data, uploaded_images)

        st.success("Registration saved successfully.")
        st.info(f"Case ID: {result['case_id']}")
        st.write(f"Images saved: {result['image_count']}")
        st.write(f"Face embeddings stored: {result['embedding_count']}")

    except ValidationError as exc:
        st.error(str(exc))
        logger.info("Registration validation failed: %s", exc)
    except Exception:
        st.error("Registration could not be completed. Please check the logs and try again.")
        logger.exception("Unexpected registration UI failure")
