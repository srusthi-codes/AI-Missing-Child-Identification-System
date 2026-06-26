import sqlite3
from typing import Any

from database.connection import database_transaction
from database.repositories.log_repository import create_activity_log
from database.repositories.record_repository import (
    SEARCH_FIELD_CASE_ID,
    SEARCH_FIELD_CHILD_NAME,
    SEARCH_FIELD_GUARDIAN_PHONE,
    fetch_registered_children,
)
from database.schema import initialize_database
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)

SEARCH_FIELD_ALL = "all"
VALID_SEARCH_FIELDS = {
    SEARCH_FIELD_ALL,
    SEARCH_FIELD_CASE_ID,
    SEARCH_FIELD_CHILD_NAME,
    SEARCH_FIELD_GUARDIAN_PHONE,
}


class DatabaseOperationError(Exception):
    """Raised when record retrieval or audit logging fails."""


def get_registered_children(
    search_field: str = SEARCH_FIELD_ALL,
    search_value: str = "",
) -> list[dict[str, Any]]:
    normalized_field, normalized_value = _validate_record_search(search_field, search_value)

    try:
        initialize_database()

        with database_transaction() as connection:
            query_field = None if normalized_field == SEARCH_FIELD_ALL else normalized_field
            records = fetch_registered_children(connection, query_field, normalized_value)

            create_activity_log(
                connection=connection,
                action="view_registered_children"
                if normalized_field == SEARCH_FIELD_ALL
                else "search_registered_children",
                entity_type="missing_child",
                details={
                    "search_field": normalized_field,
                    "has_search_value": bool(normalized_value),
                    "result_count": len(records),
                },
            )

        logger.info(
            "Retrieved registered children search_field=%s result_count=%s",
            normalized_field,
            len(records),
        )
        return records

    except sqlite3.Error as exc:
        logger.exception("Database error while retrieving registered children")
        raise DatabaseOperationError("Unable to retrieve child records from the database.") from exc


def _validate_record_search(search_field: str, search_value: str) -> tuple[str, str]:
    normalized_field = (search_field or SEARCH_FIELD_ALL).strip()
    normalized_value = " ".join(str(search_value or "").split())

    if normalized_field not in VALID_SEARCH_FIELDS:
        raise ValidationError("Invalid records search type.")

    if normalized_field == SEARCH_FIELD_ALL:
        return normalized_field, ""

    if not normalized_value:
        raise ValidationError("Enter a search value.")

    if len(normalized_value) > 120:
        raise ValidationError("Search value must be 120 characters or fewer.")

    return normalized_field, normalized_value
