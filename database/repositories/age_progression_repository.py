import sqlite3
from typing import Any


def create_age_progression_record(connection: sqlite3.Connection, record: dict[str, Any]) -> int:
    cursor = connection.execute(
        """
        INSERT INTO age_progression_history (
            child_id,
            case_id,
            source_image_path,
            generated_image_path,
            source_age,
            target_age,
            target_age_label,
            progression_years,
            model_name,
            identity_score,
            identity_quality
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["child_id"],
            record["case_id"],
            record["source_image_path"],
            record["generated_image_path"],
            record["source_age"],
            record["target_age"],
            record["target_age_label"],
            record["progression_years"],
            record["model_name"],
            record.get("identity_score"),
            record["identity_quality"],
        ),
    )
    return int(cursor.lastrowid)


def fetch_age_progression_history(
    connection: sqlite3.Connection,
    child_id: int,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    limit_sql = "LIMIT ?" if limit is not None else ""
    parameters: tuple[Any, ...] = (child_id, limit) if limit is not None else (child_id,)

    rows = connection.execute(
        f"""
        SELECT
            id AS history_id,
            child_id,
            case_id,
            source_image_path,
            generated_image_path,
            source_age,
            target_age,
            target_age_label,
            progression_years,
            model_name,
            identity_score,
            identity_quality,
            created_at
        FROM age_progression_history
        WHERE child_id = ?
        ORDER BY created_at DESC, id DESC
        {limit_sql}
        """,
        parameters,
    ).fetchall()

    return [_row_to_history(row) for row in rows]


def delete_age_progression_record(connection: sqlite3.Connection, history_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
            id AS history_id,
            generated_image_path
        FROM age_progression_history
        WHERE id = ?
        LIMIT 1
        """,
        (history_id,),
    ).fetchone()
    if row is None:
        return None

    connection.execute("DELETE FROM age_progression_history WHERE id = ?", (history_id,))
    return {"history_id": row["history_id"], "generated_image_path": row["generated_image_path"]}


def _row_to_history(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "history_id": row["history_id"],
        "child_id": row["child_id"],
        "case_id": row["case_id"],
        "source_image_path": row["source_image_path"],
        "generated_image_path": row["generated_image_path"],
        "source_age": int(row["source_age"]),
        "target_age": int(row["target_age"]),
        "target_age_label": row["target_age_label"],
        "progression_years": int(row["progression_years"]),
        "model_name": row["model_name"],
        "identity_score": None if row["identity_score"] is None else float(row["identity_score"]),
        "identity_quality": row["identity_quality"],
        "created_at": row["created_at"],
    }
