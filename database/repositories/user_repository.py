import sqlite3
from typing import Any


def create_user(connection: sqlite3.Connection, user_data: dict[str, Any]) -> int:
    cursor = connection.execute(
        """
        INSERT INTO users (
            email,
            password_hash,
            full_name,
            role,
            phone,
            is_active
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_data["email"],
            user_data["password_hash"],
            user_data["full_name"],
            user_data["role"],
            user_data.get("phone"),
            1 if user_data.get("is_active", True) else 0,
        ),
    )
    return int(cursor.lastrowid)


def fetch_user_by_email(connection: sqlite3.Connection, email: str) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
            id AS user_id,
            email,
            password_hash,
            full_name,
            role,
            phone,
            is_active,
            created_at,
            last_login_at
        FROM users
        WHERE LOWER(email) = LOWER(?)
        LIMIT 1
        """,
        (email,),
    ).fetchone()
    return None if row is None else _row_to_user(row)


def fetch_user_by_id(connection: sqlite3.Connection, user_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
            id AS user_id,
            email,
            password_hash,
            full_name,
            role,
            phone,
            is_active,
            created_at,
            last_login_at
        FROM users
        WHERE id = ?
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    return None if row is None else _row_to_user(row)


def fetch_users_by_role(connection: sqlite3.Connection, role: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            id AS user_id,
            email,
            password_hash,
            full_name,
            role,
            phone,
            is_active,
            created_at,
            last_login_at
        FROM users
        WHERE role = ?
          AND is_active = 1
        ORDER BY full_name ASC
        """,
        (role,),
    ).fetchall()
    return [_row_to_user(row) for row in rows]


def update_user_last_login(connection: sqlite3.Connection, user_id: int) -> None:
    connection.execute(
        """
        UPDATE users
        SET last_login_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (user_id,),
    )


def _row_to_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "user_id": row["user_id"],
        "email": row["email"],
        "password_hash": row["password_hash"],
        "full_name": row["full_name"],
        "role": row["role"],
        "phone": row["phone"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }
