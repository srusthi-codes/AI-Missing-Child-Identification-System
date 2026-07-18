import sqlite3
from typing import Any


def create_found_child_report(connection: sqlite3.Connection, report_data: dict[str, Any]) -> int:
    cursor = connection.execute(
        """
        INSERT INTO found_child_reports (
            reporter_user_id,
            search_id,
            uploaded_image_path,
            location,
            description,
            matches_found,
            best_similarity_score,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report_data["reporter_user_id"],
            report_data.get("search_id"),
            report_data["uploaded_image_path"],
            report_data["location"],
            report_data.get("description"),
            report_data.get("matches_found", 0),
            report_data.get("best_similarity_score", 0.0),
            report_data["status"],
        ),
    )
    return int(cursor.lastrowid)


def create_found_child_report_matches(
    connection: sqlite3.Connection,
    report_id: int,
    matches: list[dict[str, Any]],
) -> None:
    connection.executemany(
        """
        INSERT INTO found_child_report_matches (
            report_id,
            child_id,
            case_id,
            similarity_score,
            match_rank
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                report_id,
                int(match["child_id"]),
                match["case_id"],
                float(match["similarity_score"]),
                index,
            )
            for index, match in enumerate(matches, start=1)
        ],
    )


def fetch_found_child_reports(
    connection: sqlite3.Connection,
    reporter_user_id: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    where_sql = "WHERE report.reporter_user_id = ?" if reporter_user_id is not None else ""
    parameters: list[Any] = [reporter_user_id] if reporter_user_id is not None else []
    limit_sql = "LIMIT ?" if limit is not None else ""
    if limit is not None:
        parameters.append(limit)

    rows = connection.execute(
        f"""
        SELECT
            report.id AS report_id,
            report.reporter_user_id,
            user.full_name AS reporter_name,
            user.email AS reporter_email,
            report.search_id,
            report.uploaded_image_path,
            report.location,
            report.description,
            report.matches_found,
            report.best_similarity_score,
            report.status,
            report.created_at,
            report.updated_at
        FROM found_child_reports report
        INNER JOIN users user
            ON user.id = report.reporter_user_id
        {where_sql}
        ORDER BY report.created_at DESC, report.id DESC
        {limit_sql}
        """,
        parameters,
    ).fetchall()
    return [_row_to_report(row) for row in rows]


def fetch_report_matches(connection: sqlite3.Connection, report_id: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            match.id AS report_match_id,
            match.report_id,
            match.child_id,
            match.case_id,
            match.similarity_score,
            match.match_rank,
            child.full_name AS child_name,
            child.age,
            child.gender,
            child.status AS case_status,
            parent.guardian_name,
            parent.phone AS guardian_phone
        FROM found_child_report_matches match
        INNER JOIN missing_children child
            ON child.id = match.child_id
        LEFT JOIN parent_details parent
            ON parent.child_id = child.id
        WHERE match.report_id = ?
        ORDER BY match.match_rank ASC
        """,
        (report_id,),
    ).fetchall()
    return [_row_to_match(row) for row in rows]


def fetch_reports_for_child_ids(connection: sqlite3.Connection, child_ids: list[int]) -> list[dict[str, Any]]:
    if not child_ids:
        return []

    placeholders = ", ".join("?" for _ in child_ids)
    rows = connection.execute(
        f"""
        SELECT DISTINCT
            report.id AS report_id,
            report.reporter_user_id,
            user.full_name AS reporter_name,
            user.email AS reporter_email,
            report.search_id,
            report.uploaded_image_path,
            report.location,
            report.description,
            report.matches_found,
            report.best_similarity_score,
            report.status,
            report.created_at,
            report.updated_at
        FROM found_child_reports report
        INNER JOIN found_child_report_matches match
            ON match.report_id = report.id
        INNER JOIN users user
            ON user.id = report.reporter_user_id
        WHERE match.child_id IN ({placeholders})
        ORDER BY report.created_at DESC, report.id DESC
        """,
        child_ids,
    ).fetchall()
    return [_row_to_report(row) for row in rows]


def update_found_child_report_status(connection: sqlite3.Connection, report_id: int, status: str) -> bool:
    cursor = connection.execute(
        """
        UPDATE found_child_reports
        SET status = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (status, report_id),
    )
    return cursor.rowcount > 0


def _row_to_report(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "report_id": row["report_id"],
        "reporter_user_id": row["reporter_user_id"],
        "reporter_name": row["reporter_name"],
        "reporter_email": row["reporter_email"],
        "search_id": row["search_id"],
        "uploaded_image_path": row["uploaded_image_path"],
        "location": row["location"],
        "description": row["description"],
        "matches_found": int(row["matches_found"] or 0),
        "best_similarity_score": float(row["best_similarity_score"] or 0.0),
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_match(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "report_match_id": row["report_match_id"],
        "report_id": row["report_id"],
        "child_id": row["child_id"],
        "case_id": row["case_id"],
        "similarity_score": float(row["similarity_score"] or 0.0),
        "similarity_percentage": round(float(row["similarity_score"] or 0.0) * 100.0, 2),
        "match_rank": row["match_rank"],
        "child_name": row["child_name"],
        "age": row["age"],
        "gender": row["gender"],
        "case_status": row["case_status"],
        "guardian_name": row["guardian_name"],
        "guardian_phone": row["guardian_phone"],
    }
