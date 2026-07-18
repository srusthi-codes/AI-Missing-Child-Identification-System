import sqlite3
from typing import Any


def fetch_public_active_cases(connection: sqlite3.Connection, limit: int = 12) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            child.id AS child_id,
            child.case_id,
            child.full_name,
            child.age,
            child.gender,
            child.last_seen_location,
            child.last_seen_date,
            child.status,
            child.created_at AS registration_date,
            MIN(image.image_path) AS image_path
        FROM missing_children child
        LEFT JOIN child_images image
            ON image.child_id = child.id
        WHERE child.status = 'missing'
        GROUP BY child.id
        ORDER BY child.created_at DESC, child.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [_row_to_public_case(row) for row in rows]


def fetch_parent_case_summaries(
    connection: sqlite3.Connection,
    user_id: int,
    email: str,
    phone: str | None,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT DISTINCT
            child.id AS child_id,
            child.case_id,
            child.full_name,
            child.age,
            child.gender,
            child.last_seen_location,
            child.last_seen_date,
            child.status,
            child.created_at AS registration_date,
            parent.guardian_name,
            parent.phone AS guardian_phone,
            parent.email AS guardian_email,
            COUNT(DISTINCT image.id) AS uploaded_image_count
        FROM missing_children child
        LEFT JOIN parent_details parent
            ON parent.child_id = child.id
        LEFT JOIN child_images image
            ON image.child_id = child.id
        WHERE child.registered_by = ?
           OR LOWER(COALESCE(parent.email, '')) = LOWER(?)
           OR (COALESCE(parent.phone, '') != '' AND COALESCE(parent.phone, '') = COALESCE(?, ''))
        GROUP BY child.id, parent.id
        ORDER BY child.created_at DESC, child.id DESC
        """,
        (user_id, email, phone),
    ).fetchall()
    return [_row_to_parent_case(row) for row in rows]


def fetch_parent_user_recipients_for_child(connection: sqlite3.Connection, child_id: int) -> list[dict[str, Any]]:
    case_row = connection.execute(
        """
        SELECT
            child.registered_by,
            parent.email AS guardian_email,
            parent.phone AS guardian_phone
        FROM missing_children child
        LEFT JOIN parent_details parent
            ON parent.child_id = child.id
        WHERE child.id = ?
        LIMIT 1
        """,
        (child_id,),
    ).fetchone()
    if case_row is None:
        return []

    conditions: list[str] = ["role = 'parent_guardian'", "is_active = 1"]
    identity_conditions: list[str] = []
    parameters: list[Any] = []

    registered_by = case_row["registered_by"]
    if registered_by:
        identity_conditions.append("id = ?")
        parameters.append(registered_by)

    guardian_email = case_row["guardian_email"]
    if guardian_email:
        identity_conditions.append("LOWER(email) = LOWER(?)")
        parameters.append(guardian_email)

    guardian_phone = case_row["guardian_phone"]
    if guardian_phone:
        identity_conditions.append("phone = ?")
        parameters.append(guardian_phone)

    if not identity_conditions:
        return []

    rows = connection.execute(
        f"""
        SELECT
            id AS user_id,
            email,
            full_name,
            role,
            phone
        FROM users
        WHERE {' AND '.join(conditions)}
          AND ({' OR '.join(identity_conditions)})
        ORDER BY id ASC
        """,
        parameters,
    ).fetchall()
    return [_row_to_user_recipient(row) for row in rows]


def _row_to_public_case(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "child_id": row["child_id"],
        "case_id": row["case_id"],
        "full_name": row["full_name"],
        "age": row["age"],
        "gender": row["gender"],
        "last_seen_location": row["last_seen_location"],
        "last_seen_date": row["last_seen_date"],
        "status": row["status"],
        "registration_date": row["registration_date"],
        "image_path": row["image_path"],
    }


def _row_to_parent_case(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "child_id": row["child_id"],
        "case_id": row["case_id"],
        "full_name": row["full_name"],
        "age": row["age"],
        "gender": row["gender"],
        "last_seen_location": row["last_seen_location"],
        "last_seen_date": row["last_seen_date"],
        "status": row["status"],
        "registration_date": row["registration_date"],
        "guardian_name": row["guardian_name"],
        "guardian_phone": row["guardian_phone"],
        "guardian_email": row["guardian_email"],
        "uploaded_image_count": int(row["uploaded_image_count"] or 0),
    }


def _row_to_user_recipient(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "user_id": row["user_id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
        "phone": row["phone"],
    }
