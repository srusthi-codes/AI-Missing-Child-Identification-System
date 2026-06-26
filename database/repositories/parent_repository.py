import sqlite3
from typing import Any


def create_parent_details(
    connection: sqlite3.Connection,
    child_id: int,
    parent_data: dict[str, Any],
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO parent_details (
            child_id,
            guardian_name,
            relationship,
            phone,
            email,
            address,
            government_id_type,
            government_id_last4
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            child_id,
            parent_data["guardian_name"],
            parent_data["relationship"],
            parent_data["phone"],
            parent_data.get("email"),
            parent_data["address"],
            parent_data.get("government_id_type"),
            parent_data.get("government_id_last4"),
        ),
    )
    return int(cursor.lastrowid)
