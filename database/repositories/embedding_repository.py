import json
import sqlite3
from typing import Any


def create_face_embedding_records(
    connection: sqlite3.Connection,
    child_id: int,
    embedding_records: list[dict[str, Any]],
) -> None:
    connection.executemany(
        """
        INSERT INTO face_embeddings (
            child_id,
            image_path,
            image_hash,
            embedding,
            model_name,
            detector_backend,
            embedding_dimension,
            quality_score,
            face_confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                child_id,
                record["image_path"],
                record["image_hash"],
                json.dumps(record["embedding"], ensure_ascii=True),
                record["model_name"],
                record["detector_backend"],
                record["embedding_dimension"],
                record["quality_score"],
                record.get("face_confidence"),
            )
            for record in embedding_records
        ],
    )


def fetch_all_embedding_candidates(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            embedding.id AS embedding_id,
            embedding.child_id,
            embedding.image_path,
            embedding.image_hash,
            embedding.embedding,
            embedding.model_name,
            embedding.detector_backend,
            embedding.embedding_dimension,
            embedding.quality_score,
            embedding.face_confidence,
            child.case_id,
            child.full_name,
            child.age,
            child.gender,
            parent.guardian_name,
            parent.phone AS guardian_phone
        FROM face_embeddings embedding
        INNER JOIN missing_children child
            ON child.id = embedding.child_id
        LEFT JOIN parent_details parent
            ON parent.child_id = child.id
        ORDER BY embedding.created_at DESC, embedding.id DESC
        """
    ).fetchall()

    return [_row_to_candidate(row) for row in rows]


def _row_to_candidate(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "embedding_id": row["embedding_id"],
        "child_id": row["child_id"],
        "image_path": row["image_path"],
        "image_hash": row["image_hash"],
        "embedding": row["embedding"],
        "model_name": row["model_name"],
        "detector_backend": row["detector_backend"],
        "embedding_dimension": row["embedding_dimension"],
        "quality_score": row["quality_score"],
        "face_confidence": row["face_confidence"],
        "case_id": row["case_id"],
        "full_name": row["full_name"],
        "age": row["age"],
        "gender": row["gender"],
        "guardian_name": row["guardian_name"],
        "guardian_phone": row["guardian_phone"],
    }
