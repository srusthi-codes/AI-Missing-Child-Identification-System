import sqlite3
from typing import Any


def create_notification(connection: sqlite3.Connection, notification_data: dict[str, Any]) -> int:
    cursor = connection.execute(
        """
        INSERT INTO notifications (
            recipient_user_id,
            recipient_role,
            child_id,
            report_id,
            case_id,
            title,
            message,
            notification_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            notification_data.get("recipient_user_id"),
            notification_data.get("recipient_role"),
            notification_data.get("child_id"),
            notification_data.get("report_id"),
            notification_data.get("case_id"),
            notification_data["title"],
            notification_data["message"],
            notification_data["notification_type"],
        ),
    )
    return int(cursor.lastrowid)


def fetch_notifications_for_user(
    connection: sqlite3.Connection,
    user_id: int,
    role: str,
    child_ids: list[int] | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    parameters: list[Any] = [user_id]
    conditions = ["notification.recipient_user_id = ?"]

    if role == "authority":
        parameters.append(role)
        conditions.append("notification.recipient_user_id IS NULL AND notification.recipient_role = ?")
    elif role and child_ids:
        placeholders = ", ".join("?" for _ in child_ids)
        parameters.append(role)
        parameters.extend(child_ids)
        conditions.append(
            f"notification.recipient_user_id IS NULL "
            f"AND notification.recipient_role = ? "
            f"AND notification.child_id IN ({placeholders})"
        )

    parameters.append(limit)
    rows = connection.execute(
        f"""
        SELECT
            notification.id AS notification_id,
            notification.recipient_user_id,
            notification.recipient_role,
            notification.child_id,
            notification.report_id,
            notification.case_id,
            notification.title,
            notification.message,
            notification.notification_type,
            notification.is_read,
            notification.created_at
        FROM notifications notification
        WHERE {' OR '.join(f'({condition})' for condition in conditions)}
        ORDER BY notification.created_at DESC, notification.id DESC
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    return [_row_to_notification(row) for row in rows]


def count_unread_notifications_for_user(
    connection: sqlite3.Connection,
    user_id: int,
    role: str,
    child_ids: list[int] | None = None,
) -> int:
    parameters: list[Any] = [user_id]
    conditions = ["recipient_user_id = ?"]

    if role == "authority":
        parameters.append(role)
        conditions.append("recipient_user_id IS NULL AND recipient_role = ?")
    elif role and child_ids:
        placeholders = ", ".join("?" for _ in child_ids)
        parameters.append(role)
        parameters.extend(child_ids)
        conditions.append(f"recipient_user_id IS NULL AND recipient_role = ? AND child_id IN ({placeholders})")

    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM notifications
        WHERE is_read = 0
          AND ({' OR '.join(f'({condition})' for condition in conditions)})
        """,
        parameters,
    ).fetchone()
    return int(row[0] if row else 0)


def mark_notification_read(connection: sqlite3.Connection, notification_id: int, user_id: int) -> bool:
    cursor = connection.execute(
        """
        UPDATE notifications
        SET is_read = 1
        WHERE id = ?
          AND (recipient_user_id = ? OR recipient_user_id IS NULL)
        """,
        (notification_id, user_id),
    )
    return cursor.rowcount > 0


def _row_to_notification(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "notification_id": row["notification_id"],
        "recipient_user_id": row["recipient_user_id"],
        "recipient_role": row["recipient_role"],
        "child_id": row["child_id"],
        "report_id": row["report_id"],
        "case_id": row["case_id"],
        "title": row["title"],
        "message": row["message"],
        "notification_type": row["notification_type"],
        "is_read": bool(row["is_read"]),
        "created_at": row["created_at"],
    }
