import sqlite3
from contextlib import contextmanager
from typing import Iterator

from config.settings import DATABASE_DIR, DATABASE_PATH, SQLITE_TIMEOUT_SECONDS


def get_connection() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=SQLITE_TIMEOUT_SECONDS,
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def database_transaction() -> Iterator[sqlite3.Connection]:
    connection = get_connection()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
