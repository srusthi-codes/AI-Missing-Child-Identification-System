import json
import sqlite3
from typing import Any


def create_activity_log(
    connection: sqlite3.Connection,
    action: str,
    entity_type: str | None = None,
    entity_id: int | None = None,
    user_id: int | None = None,
    ip_address: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    safe_details = json.dumps(details or {}, ensure_ascii=True)
    connection.execute(
        """
        INSERT INTO activity_logs (
            user_id,
            action,
            entity_type,
            entity_id,
            ip_address,
            details
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, action, entity_type, entity_id, ip_address, safe_details),
    )
