import sqlite3
from typing import Any


def create_child_image_records(
    connection: sqlite3.Connection,
    child_id: int,
    case_id: str,
    image_records: list[dict[str, Any]],
) -> None:
    connection.executemany(
        """
        INSERT INTO child_images (
            child_id,
            case_id,
            image_path,
            original_filename,
            content_type,
            file_size,
            image_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                child_id,
                case_id,
                record["image_path"],
                record["original_filename"],
                record["content_type"],
                record["file_size"],
                record["image_hash"],
            )
            for record in image_records
        ],
    )
