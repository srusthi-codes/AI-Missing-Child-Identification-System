import csv
import io
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

from config.constants import GENDER_OPTIONS, MAX_CHILD_AGE, MIN_CHILD_AGE
from config.settings import BASE_DIR
from database.connection import database_transaction
from database.repositories.dashboard_repository import (
    delete_child_case as delete_child_case_record,
    fetch_case_details,
    fetch_dashboard_statistics,
    fetch_filtered_child_records,
    fetch_recent_registrations,
    fetch_search_history,
)
from database.repositories.log_repository import create_activity_log
from database.schema import initialize_database
from utils.file_handler import cleanup_saved_files
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


class DashboardDatabaseError(Exception):
    """Raised when dashboard data cannot be loaded or changed."""


def get_dashboard_overview() -> dict[str, Any]:
    try:
        initialize_database()
        with database_transaction() as connection:
            statistics = fetch_dashboard_statistics(connection)
            recent_registrations = fetch_recent_registrations(connection, limit=10)
            recent_search_history = fetch_search_history(connection, limit=10)
            create_activity_log(
                connection=connection,
                action="view_admin_dashboard",
                entity_type="dashboard",
                details={
                    "recent_registration_count": len(recent_registrations),
                    "recent_search_history_count": len(recent_search_history),
                },
            )

        logger.info("Loaded admin dashboard overview")
        return {
            "statistics": statistics,
            "recent_registrations": recent_registrations,
            "recent_search_history": recent_search_history,
        }
    except sqlite3.Error as exc:
        logger.exception("Database error while loading dashboard overview")
        raise DashboardDatabaseError("Unable to load dashboard data.") from exc


def search_dashboard_child_records(filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    normalized_filters = _normalize_child_filters(filters or {})

    try:
        initialize_database()
        with database_transaction() as connection:
            records = fetch_filtered_child_records(connection, normalized_filters)
            create_activity_log(
                connection=connection,
                action="filter_dashboard_child_records",
                entity_type="missing_child",
                details={
                    "active_filters": sorted(
                        key for key, value in normalized_filters.items() if value not in (None, "")
                    ),
                    "result_count": len(records),
                },
            )

        logger.info("Loaded dashboard child records result_count=%s", len(records))
        return records
    except sqlite3.Error as exc:
        logger.exception("Database error while filtering dashboard child records")
        raise DashboardDatabaseError("Unable to load filtered child records.") from exc


def get_child_case_details(child_id: int) -> dict[str, Any]:
    normalized_child_id = _validate_child_id(child_id)

    try:
        initialize_database()
        with database_transaction() as connection:
            case_details = fetch_case_details(connection, normalized_child_id)
            if case_details is None:
                raise ValidationError("Selected case was not found.")

            create_activity_log(
                connection=connection,
                action="view_dashboard_case_details",
                entity_type="missing_child",
                entity_id=normalized_child_id,
                details={"case_id": case_details["case_id"]},
            )

        logger.info("Loaded case details child_id=%s", normalized_child_id)
        return case_details
    except ValidationError:
        raise
    except sqlite3.Error as exc:
        logger.exception("Database error while loading case details child_id=%s", normalized_child_id)
        raise DashboardDatabaseError("Unable to load case details.") from exc


def delete_child_case(child_id: int, confirmation_case_id: str) -> dict[str, Any]:
    normalized_child_id = _validate_child_id(child_id)
    normalized_confirmation = " ".join(str(confirmation_case_id or "").split())

    try:
        initialize_database()
        with database_transaction() as connection:
            case_details = fetch_case_details(connection, normalized_child_id)
            if case_details is None:
                raise ValidationError("Selected case was not found.")

            if normalized_confirmation != case_details["case_id"]:
                raise ValidationError("Confirmation Case ID does not match the selected case.")

            image_paths = [_absolute_path(image["image_path"]) for image in case_details["images"]]
            deleted = delete_child_case_record(connection, normalized_child_id)
            if not deleted:
                raise ValidationError("Selected case was not found.")

            create_activity_log(
                connection=connection,
                action="delete_missing_child_case",
                entity_type="missing_child",
                entity_id=normalized_child_id,
                details={
                    "case_id": case_details["case_id"],
                    "child_name": case_details["full_name"],
                    "deleted_image_count": len(image_paths),
                    "deleted_embedding_count": case_details["embedding_count"],
                },
            )

        cleanup_saved_files([str(path) for path in image_paths])
        logger.warning(
            "Deleted missing child case child_id=%s case_id=%s",
            normalized_child_id,
            case_details["case_id"],
        )
        return {
            "deleted": True,
            "case_id": case_details["case_id"],
            "child_name": case_details["full_name"],
            "deleted_image_count": len(image_paths),
            "deleted_embedding_count": case_details["embedding_count"],
        }
    except ValidationError:
        raise
    except sqlite3.Error as exc:
        logger.exception("Database error while deleting child case child_id=%s", normalized_child_id)
        raise DashboardDatabaseError("Unable to delete selected case.") from exc


def export_child_records_csv(filters: dict[str, Any] | None = None) -> bytes:
    records = search_dashboard_child_records(filters or {})
    columns = [
        "case_id",
        "full_name",
        "age",
        "gender",
        "status",
        "guardian_name",
        "guardian_phone",
        "uploaded_image_count",
        "embedding_count",
        "registration_date",
    ]
    return _records_to_csv_bytes(records, columns)


def export_search_history_csv() -> bytes:
    try:
        initialize_database()
        with database_transaction() as connection:
            records = fetch_search_history(connection, limit=None)
            create_activity_log(
                connection=connection,
                action="export_search_history_csv",
                entity_type="search_history",
                details={"exported_count": len(records)},
            )

        return _records_to_csv_bytes(
            records,
            [
                "search_id",
                "uploaded_image_path",
                "matches_found",
                "best_similarity_score",
                "status",
                "error_message",
                "created_at",
            ],
        )
    except sqlite3.Error as exc:
        logger.exception("Database error while exporting search history")
        raise DashboardDatabaseError("Unable to export search history.") from exc


def get_recent_search_history(limit: int = 10) -> list[dict[str, Any]]:
    try:
        initialize_database()
        with database_transaction() as connection:
            records = fetch_search_history(connection, limit=limit)
        return records
    except sqlite3.Error as exc:
        logger.exception("Database error while retrieving search history")
        raise DashboardDatabaseError("Unable to load search history.") from exc


def get_recent_registrations(limit: int = 10) -> list[dict[str, Any]]:
    try:
        initialize_database()
        with database_transaction() as connection:
            records = fetch_recent_registrations(connection, limit=limit)
        return records
    except sqlite3.Error as exc:
        logger.exception("Database error while retrieving recent registrations")
        raise DashboardDatabaseError("Unable to load recent registrations.") from exc


def _normalize_child_filters(filters: dict[str, Any]) -> dict[str, Any]:
    case_id = _optional_text(filters.get("case_id"), "Case ID", 80)
    child_name = _optional_text(filters.get("child_name"), "Child name", 120)
    gender = _optional_text(filters.get("gender"), "Gender", 20)
    if gender and gender not in GENDER_OPTIONS:
        raise ValidationError("Gender filter has an invalid value.")

    age = _optional_age(filters.get("age"))
    registration_date = _optional_registration_date(filters.get("registration_date"))

    return {
        "case_id": case_id,
        "child_name": child_name,
        "gender": gender,
        "age": age,
        "registration_date": registration_date,
    }


def _optional_text(value: Any, field_name: str, max_length: int) -> str | None:
    cleaned = " ".join(str(value or "").split())
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        raise ValidationError(f"{field_name} must be {max_length} characters or fewer.")
    return cleaned


def _optional_age(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        age = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Age filter must be a valid number.") from exc
    if age < MIN_CHILD_AGE or age > MAX_CHILD_AGE:
        raise ValidationError(f"Age filter must be between {MIN_CHILD_AGE} and {MAX_CHILD_AGE}.")
    return age


def _optional_registration_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date().isoformat()
        except ValueError as exc:
            raise ValidationError("Registration date filter must use YYYY-MM-DD format.") from exc
    raise ValidationError("Registration date filter has an invalid value.")


def _validate_child_id(value: Any) -> int:
    try:
        child_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Invalid child case selection.") from exc
    if child_id <= 0:
        raise ValidationError("Invalid child case selection.")
    return child_id


def _absolute_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _records_to_csv_bytes(records: list[dict[str, Any]], columns: list[str]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({column: _csv_value(record.get(column)) for column in columns})
    return output.getvalue().encode("utf-8")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value
