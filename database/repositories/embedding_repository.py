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
