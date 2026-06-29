import sqlite3
from typing import Any


def fetch_dashboard_statistics(connection: sqlite3.Connection) -> dict[str, Any]:
    total_children = _count(connection, "SELECT COUNT(*) FROM missing_children")
    total_embeddings = _count(connection, "SELECT COUNT(*) FROM face_embeddings")
    total_search_requests = _count(connection, "SELECT COUNT(*) FROM search_history")
    successful_matches = _count(
        connection,
        """
        SELECT COUNT(*)
        FROM search_history
        WHERE status = 'completed'
          AND matches_found > 0
        """,
    )
    unsuccessful_searches = max(total_search_requests - successful_matches, 0)
    match_success_percentage = (
        round((successful_matches / total_search_requests) * 100.0, 2)
        if total_search_requests
        else 0.0
    )

    return {
        "total_registered_children": total_children,
        "total_face_embeddings": total_embeddings,
        "total_search_requests": total_search_requests,
        "total_successful_matches": successful_matches,
        "total_unsuccessful_searches": unsuccessful_searches,
        "match_success_percentage": match_success_percentage,
    }


def fetch_recent_registrations(connection: sqlite3.Connection, limit: int = 10) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            child.id AS child_id,
            child.case_id,
            child.full_name,
            child.age,
            child.gender,
            child.status,
            child.created_at AS registration_date,
            parent.guardian_name,
            parent.phone AS guardian_phone,
            COUNT(DISTINCT image.id) AS uploaded_image_count,
            COUNT(DISTINCT embedding.id) AS embedding_count
        FROM missing_children child
        LEFT JOIN parent_details parent
            ON parent.child_id = child.id
        LEFT JOIN child_images image
            ON image.child_id = child.id
        LEFT JOIN face_embeddings embedding
            ON embedding.child_id = child.id
        GROUP BY child.id, parent.id
        ORDER BY child.created_at DESC, child.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    return [_row_to_child_summary(row) for row in rows]


def fetch_filtered_child_records(
    connection: sqlite3.Connection,
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    where_clauses: list[str] = []
    parameters: list[Any] = []

    case_id = filters.get("case_id")
    if case_id:
        where_clauses.append("child.case_id LIKE ? ESCAPE '\\'")
        parameters.append(_like_pattern(case_id))

    child_name = filters.get("child_name")
    if child_name:
        where_clauses.append("child.full_name LIKE ? ESCAPE '\\'")
        parameters.append(_like_pattern(child_name))

    gender = filters.get("gender")
    if gender:
        where_clauses.append("child.gender = ?")
        parameters.append(gender)

    age = filters.get("age")
    if age is not None:
        where_clauses.append("child.age = ?")
        parameters.append(age)

    registration_date = filters.get("registration_date")
    if registration_date:
        where_clauses.append("DATE(child.created_at) = ?")
        parameters.append(registration_date)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    rows = connection.execute(
        f"""
        SELECT
            child.id AS child_id,
            child.case_id,
            child.full_name,
            child.age,
            child.gender,
            child.status,
            child.created_at AS registration_date,
            parent.guardian_name,
            parent.phone AS guardian_phone,
            COUNT(DISTINCT image.id) AS uploaded_image_count,
            COUNT(DISTINCT embedding.id) AS embedding_count
        FROM missing_children child
        LEFT JOIN parent_details parent
            ON parent.child_id = child.id
        LEFT JOIN child_images image
            ON image.child_id = child.id
        LEFT JOIN face_embeddings embedding
            ON embedding.child_id = child.id
        {where_sql}
        GROUP BY child.id, parent.id
        ORDER BY child.created_at DESC, child.id DESC
        """,
        parameters,
    ).fetchall()

    return [_row_to_child_summary(row) for row in rows]


def fetch_case_details(connection: sqlite3.Connection, child_id: int) -> dict[str, Any] | None:
    case_row = connection.execute(
        """
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
            parent.government_id_last4
        FROM missing_children child
        LEFT JOIN parent_details parent
            ON parent.child_id = child.id
        WHERE child.id = ?
        LIMIT 1
        """,
        (child_id,),
    ).fetchone()

    if case_row is None:
        return None

    image_rows = connection.execute(
        """
        SELECT
            id AS image_id,
            image_path,
            original_filename,
            content_type,
            file_size,
            image_hash,
            created_at
        FROM child_images
        WHERE child_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (child_id,),
    ).fetchall()

    embedding_rows = connection.execute(
        """
        SELECT
            id AS embedding_id,
            image_path,
            image_hash,
            model_name,
            detector_backend,
            embedding_dimension,
            quality_score,
            face_confidence,
            created_at
        FROM face_embeddings
        WHERE child_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (child_id,),
    ).fetchall()

    case_details = _row_to_case_details(case_row)
    case_details["images"] = [_row_to_image(row) for row in image_rows]
    case_details["embeddings"] = [_row_to_embedding_summary(row) for row in embedding_rows]
    case_details["uploaded_image_count"] = len(case_details["images"])
    case_details["embedding_count"] = len(case_details["embeddings"])
    case_details["embedding_status"] = "Stored" if case_details["embedding_count"] else "Missing"
    return case_details


def fetch_search_history(connection: sqlite3.Connection, limit: int | None = None) -> list[dict[str, Any]]:
    limit_sql = "LIMIT ?" if limit is not None else ""
    parameters: tuple[Any, ...] = (limit,) if limit is not None else ()

    rows = connection.execute(
        f"""
        SELECT
            id AS search_id,
            uploaded_image_path,
            image_hash,
            matches_found,
            best_similarity_score,
            status,
            error_message,
            created_at
        FROM search_history
        ORDER BY created_at DESC, id DESC
        {limit_sql}
        """,
        parameters,
    ).fetchall()

    return [_row_to_search_history(row) for row in rows]


def delete_child_case(connection: sqlite3.Connection, child_id: int) -> bool:
    cursor = connection.execute(
        "DELETE FROM missing_children WHERE id = ?",
        (child_id,),
    )
    return cursor.rowcount > 0


def _count(connection: sqlite3.Connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    return int(row[0] if row else 0)


def _like_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _row_to_child_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "child_id": row["child_id"],
        "case_id": row["case_id"],
        "full_name": row["full_name"],
        "age": row["age"],
        "gender": row["gender"],
        "status": row["status"],
        "registration_date": row["registration_date"],
        "guardian_name": row["guardian_name"],
        "guardian_phone": row["guardian_phone"],
        "uploaded_image_count": int(row["uploaded_image_count"] or 0),
        "embedding_count": int(row["embedding_count"] or 0),
    }


def _row_to_case_details(row: sqlite3.Row) -> dict[str, Any]:
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
    }


def _row_to_image(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "original_filename": row["original_filename"],
        "content_type": row["content_type"],
        "file_size": row["file_size"],
        "image_hash": row["image_hash"],
        "created_at": row["created_at"],
    }


def _row_to_embedding_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "embedding_id": row["embedding_id"],
        "image_path": row["image_path"],
        "image_hash": row["image_hash"],
        "model_name": row["model_name"],
        "detector_backend": row["detector_backend"],
        "embedding_dimension": row["embedding_dimension"],
        "quality_score": row["quality_score"],
        "face_confidence": row["face_confidence"],
        "created_at": row["created_at"],
    }


def _row_to_search_history(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "search_id": row["search_id"],
        "uploaded_image_path": row["uploaded_image_path"],
        "image_hash": row["image_hash"],
        "matches_found": int(row["matches_found"] or 0),
        "best_similarity_score": float(row["best_similarity_score"] or 0.0),
        "status": row["status"],
        "error_message": row["error_message"],
        "created_at": row["created_at"],
    }
