import re
from datetime import date, datetime, time
from typing import Any

from config.constants import (
    GENDER_OPTIONS,
    MAX_ADDRESS_LENGTH,
    MAX_CHILD_AGE,
    MAX_TEXT_LENGTH,
    MIN_CHILD_AGE,
    RELATIONSHIP_OPTIONS,
)


class ValidationError(Exception):
    pass


def validate_child_data(data: dict[str, Any]) -> dict[str, Any]:
    full_name = _required_text(data.get("full_name"), "Child full name", max_length=120)
    age = _validate_age(data.get("age"))
    gender = _choice(data.get("gender"), "Gender", GENDER_OPTIONS)
    identification_marks = _optional_text(
        data.get("identification_marks"),
        "Identification marks",
        max_length=MAX_TEXT_LENGTH,
    )
    last_seen_location = _required_text(
        data.get("last_seen_location"),
        "Last seen location",
        max_length=200,
    )
    last_seen_date = _validate_last_seen_date(data.get("last_seen_date"))
    last_seen_time = _validate_optional_time(data.get("last_seen_time"))
    description = _optional_text(data.get("description"), "Description", max_length=MAX_TEXT_LENGTH)

    return {
        "full_name": full_name,
        "age": age,
        "gender": gender,
        "identification_marks": identification_marks,
        "last_seen_location": last_seen_location,
        "last_seen_date": last_seen_date,
        "last_seen_time": last_seen_time,
        "description": description,
    }


def validate_parent_data(data: dict[str, Any]) -> dict[str, Any]:
    guardian_name = _required_text(data.get("guardian_name"), "Guardian name", max_length=120)
    relationship = _choice(data.get("relationship"), "Relationship", RELATIONSHIP_OPTIONS)
    phone = _validate_phone(data.get("phone"))
    email = _validate_optional_email(data.get("email"))
    address = _required_text(data.get("address"), "Address", max_length=MAX_ADDRESS_LENGTH)
    government_id_type = _optional_text(data.get("government_id_type"), "Government ID type", max_length=60)
    government_id_last4 = _validate_government_id_last4(data.get("government_id_last4"))

    return {
        "guardian_name": guardian_name,
        "relationship": relationship,
        "phone": phone,
        "email": email,
        "address": address,
        "government_id_type": government_id_type,
        "government_id_last4": government_id_last4,
    }


def _required_text(value: Any, field_name: str, max_length: int) -> str:
    cleaned = _normalize_text(value)
    if not cleaned:
        raise ValidationError(f"{field_name} is required.")
    if len(cleaned) > max_length:
        raise ValidationError(f"{field_name} must be {max_length} characters or fewer.")
    return cleaned


def _optional_text(value: Any, field_name: str, max_length: int) -> str | None:
    cleaned = _normalize_text(value)
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        raise ValidationError(f"{field_name} must be {max_length} characters or fewer.")
    return cleaned


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _validate_age(value: Any) -> int:
    try:
        age = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Age must be a valid number.") from exc

    if age < MIN_CHILD_AGE or age > MAX_CHILD_AGE:
        raise ValidationError(f"Age must be between {MIN_CHILD_AGE} and {MAX_CHILD_AGE}.")
    return age


def _choice(value: Any, field_name: str, choices: list[str]) -> str:
    cleaned = _required_text(value, field_name, max_length=80)
    if cleaned not in choices:
        raise ValidationError(f"{field_name} has an invalid value.")
    return cleaned


def _validate_last_seen_date(value: Any) -> str:
    if isinstance(value, datetime):
        parsed_date = value.date()
    elif isinstance(value, date):
        parsed_date = value
    elif isinstance(value, str):
        try:
            parsed_date = datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValidationError("Last seen date must use YYYY-MM-DD format.") from exc
    else:
        raise ValidationError("Last seen date is required.")

    if parsed_date > date.today():
        raise ValidationError("Last seen date cannot be in the future.")
    return parsed_date.isoformat()


def _validate_optional_time(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%H:%M:%S").time().strftime("%H:%M:%S")
        except ValueError as exc:
            raise ValidationError("Last seen time must use HH:MM:SS format.") from exc
    raise ValidationError("Last seen time has an invalid value.")


def _validate_phone(value: Any) -> str:
    phone = _required_text(value, "Phone number", max_length=20)
    if not re.fullmatch(r"[+\d][\d\s().-]{6,19}", phone):
        raise ValidationError("Phone number is invalid.")

    digit_count = len(re.sub(r"\D", "", phone))
    if digit_count < 7 or digit_count > 15:
        raise ValidationError("Phone number must contain 7 to 15 digits.")
    return phone


def _validate_optional_email(value: Any) -> str | None:
    email = _optional_text(value, "Email address", max_length=120)
    if email is None:
        return None
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValidationError("Email address is invalid.")
    return email.lower()


def _validate_government_id_last4(value: Any) -> str | None:
    last4 = _optional_text(value, "Government ID last 4 digits", max_length=4)
    if last4 is None:
        return None
    if not re.fullmatch(r"\d{4}", last4):
        raise ValidationError("Government ID last 4 digits must contain exactly 4 digits.")
    return last4
