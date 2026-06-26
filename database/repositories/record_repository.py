import re
import sqlite3
from typing import Any


SEARCH_FIELD_CASE_ID = "case_id"
SEARCH_FIELD_CHILD_NAME = "child_name"
SEARCH_FIELD_GUARDIAN_PHONE = "guardian_phone"


def fetch_registered_children(
    connection: sqlite3.Connection,
    search_field: str | None = None,
    search_value: str | None = None,
) -> list[dict[str, Any]]:
    where_clause = ""
    parameters: list[str] = []

    if search_field and search_value:
        where_clause, parameters = _build_search_clause(search_field, search_value)

    rows = connection.execute(
        f"""
        SELECT
            child.id AS child_id,
            child.case_id,
            child.full_name,
            child.age,
            child.gender,
            child.identification_marks,
            child.last_seen_location,
            child.last_seen_date,
            child.last_seen_time,
            child.description,
            child.status,
            child.created_at AS registration_date,
            child.updated_at,
            parent.id AS parent_id,
            parent.guardian_name,
            parent.relationship,
            parent.phone AS guardian_phone,
            parent.email AS guardian_email,
            parent.address AS guardian_address,
            parent.government_id_type,
            parent.government_id_last4,
            COUNT(image.id) AS uploaded_image_count
        FROM missing_children child
        LEFT JOIN parent_details parent
            ON parent.child_id = child.id
        LEFT JOIN child_images image
            ON image.child_id = child.id
        {where_clause}
        GROUP BY child.id, parent.id
        ORDER BY child.created_at DESC, child.id DESC
        """,
        parameters,
    ).fetchall()

    return [_row_to_record(row) for row in rows]


def _build_search_clause(search_field: str, search_value: str) -> tuple[str, list[str]]:
    like_value = _like_pattern(search_value)

    if search_field == SEARCH_FIELD_CASE_ID:
        return "WHERE child.case_id LIKE ? ESCAPE '\\'", [like_value]

    if search_field == SEARCH_FIELD_CHILD_NAME:
        return "WHERE child.full_name LIKE ? ESCAPE '\\'", [like_value]

    if search_field == SEARCH_FIELD_GUARDIAN_PHONE:
        digits_only = re.sub(r"\D", "", search_value)
        if digits_only:
            normalized_phone_expression = """
                REPLACE(
                    REPLACE(
                        REPLACE(
                            REPLACE(
                                REPLACE(parent.phone, ' ', ''),
                            '-', ''),
                        '(', ''),
                    ')', ''),
                '+', '')
            """
            return (
                f"""
                WHERE parent.phone LIKE ? ESCAPE '\\'
                   OR {normalized_phone_expression} LIKE ? ESCAPE '\\'
                """,
                [like_value, _like_pattern(digits_only)],
            )
        return "WHERE parent.phone LIKE ? ESCAPE '\\'", [like_value]

    raise ValueError("Unsupported search field")


def _like_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "child_id": row["child_id"],
        "case_id": row["case_id"],
        "full_name": row["full_name"],
        "age": row["age"],
        "gender": row["gender"],
        "identification_marks": row["identification_marks"],
        "last_seen_location": row["last_seen_location"],
        "last_seen_date": row["last_seen_date"],
        "last_seen_time": row["last_seen_time"],
        "description": row["description"],
        "status": row["status"],
        "registration_date": row["registration_date"],
        "updated_at": row["updated_at"],
        "parent_id": row["parent_id"],
        "guardian_name": row["guardian_name"],
        "relationship": row["relationship"],
        "guardian_phone": row["guardian_phone"],
        "guardian_email": row["guardian_email"],
        "guardian_address": row["guardian_address"],
        "government_id_type": row["government_id_type"],
        "government_id_last4": row["government_id_last4"],
        "uploaded_image_count": int(row["uploaded_image_count"] or 0),
    }
