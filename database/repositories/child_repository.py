import sqlite3
from typing import Any


def create_missing_child(connection: sqlite3.Connection, child_data: dict[str, Any]) -> int:
    cursor = connection.execute(
        """
        INSERT INTO missing_children (
            case_id,
            full_name,
            age,
            gender,
            identification_marks,
            last_seen_location,
            last_seen_date,
            last_seen_time,
            description,
            status,
            registered_by
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            child_data["case_id"],
            child_data["full_name"],
            child_data["age"],
            child_data["gender"],
            child_data.get("identification_marks"),
            child_data["last_seen_location"],
            child_data["last_seen_date"],
            child_data.get("last_seen_time"),
            child_data.get("description"),
            child_data["status"],
            child_data.get("registered_by"),
        ),
    )
    return int(cursor.lastrowid)


def case_id_exists(connection: sqlite3.Connection, case_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM missing_children WHERE case_id = ? LIMIT 1",
        (case_id,),
    ).fetchone()
    return row is not None
