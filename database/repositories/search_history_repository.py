import sqlite3
from typing import Any


def create_search_history_record(
    connection: sqlite3.Connection,
    search_data: dict[str, Any],
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO search_history (
            uploaded_image_path,
            image_hash,
            matches_found,
            best_similarity_score,
            status,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            search_data["uploaded_image_path"],
            search_data.get("image_hash"),
            search_data.get("matches_found", 0),
            search_data.get("best_similarity_score", 0.0),
            search_data.get("status", "completed"),
            search_data.get("error_message"),
        ),
    )
    return int(cursor.lastrowid)


def count_searches_by_image_hash(connection: sqlite3.Connection, image_hash: str) -> int:
    row = connection.execute(
        """
        SELECT COUNT(*)
        FROM search_history
        WHERE image_hash = ?
        """,
        (image_hash,),
    ).fetchone()
    return int(row[0] if row else 0)
