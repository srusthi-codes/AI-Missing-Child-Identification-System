import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

from config.constants import (
    FOUND_REPORT_STATUS_DISMISSED,
    FOUND_REPORT_STATUS_NO_MATCH,
    FOUND_REPORT_STATUS_PENDING,
    FOUND_REPORT_STATUS_VERIFIED,
    FOUND_REPORT_STATUSES,
    ROLE_AUTHORITY,
    ROLE_FINDER,
    ROLE_PARENT,
)
from config.settings import BASE_DIR
from database.connection import database_transaction
from database.repositories.dashboard_repository import fetch_case_details, fetch_dashboard_statistics
from database.repositories.found_report_repository import (
    create_found_child_report,
    create_found_child_report_matches,
    fetch_found_child_reports,
    fetch_report_matches,
    fetch_reports_for_child_ids,
    update_found_child_report_status,
)
from database.repositories.log_repository import create_activity_log
from database.repositories.notification_repository import (
    create_notification,
    fetch_notifications_for_user,
    mark_notification_read,
)
from database.repositories.role_dashboard_repository import (
    fetch_parent_case_summaries,
    fetch_parent_user_recipients_for_child,
    fetch_public_active_cases,
)
from database.schema import initialize_database
from services.dashboard_service import get_dashboard_overview
from services.matching_service import MatchingDatabaseError, search_found_child
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


class RoleDashboardError(Exception):
    """Raised when a role dashboard operation fails."""


def get_public_home_overview() -> dict[str, Any]:
    try:
        initialize_database()
        with database_transaction() as connection:
            statistics = fetch_dashboard_statistics(connection)
            active_cases = fetch_public_active_cases(connection, limit=12)
        for case in active_cases:
            case["missing_days"] = calculate_missing_days(case.get("last_seen_date"))
            case["absolute_image_path"] = _absolute_path(case.get("image_path"))
        return {"statistics": statistics, "active_cases": active_cases}
    except sqlite3.Error as exc:
        logger.exception("Database error while loading public home overview")
        raise RoleDashboardError("Unable to load public case overview.") from exc


def get_parent_dashboard(user: dict[str, Any]) -> dict[str, Any]:
    _require_role(user, ROLE_PARENT)
    try:
        initialize_database()
        with database_transaction() as connection:
            cases = fetch_parent_case_summaries(
                connection,
                user_id=int(user["user_id"]),
                email=user["email"],
                phone=user.get("phone"),
            )
            case_details = [fetch_case_details(connection, int(case["child_id"])) for case in cases]
            child_ids = [int(case["child_id"]) for case in cases]
            reports = fetch_reports_for_child_ids(connection, child_ids)
            notifications = fetch_notifications_for_user(
                connection,
                user_id=int(user["user_id"]),
                role=ROLE_PARENT,
                child_ids=child_ids,
                limit=25,
            )
            create_activity_log(
                connection=connection,
                action="view_parent_dashboard",
                entity_type="users",
                entity_id=int(user["user_id"]),
                user_id=int(user["user_id"]),
                details={"case_count": len(cases), "notification_count": len(notifications)},
            )

        enriched_cases = []
        for details in case_details:
            if details is None:
                continue
            details["missing_days"] = calculate_missing_days(details.get("last_seen_date"))
            for image in details.get("images", []):
                image["absolute_image_path"] = _absolute_path(image["image_path"])
            enriched_cases.append(details)

        return {
            "cases": enriched_cases,
            "matched_reports": reports,
            "notifications": notifications,
        }
    except sqlite3.Error as exc:
        logger.exception("Database error while loading parent dashboard user_id=%s", user.get("user_id"))
        raise RoleDashboardError("Unable to load parent dashboard.") from exc


def get_finder_dashboard(user: dict[str, Any]) -> dict[str, Any]:
    _require_role(user, ROLE_FINDER)
    try:
        initialize_database()
        with database_transaction() as connection:
            reports = fetch_found_child_reports(connection, reporter_user_id=int(user["user_id"]), limit=25)
            for report in reports:
                report["matches"] = fetch_report_matches(connection, int(report["report_id"]))
                report["absolute_image_path"] = _absolute_path(report["uploaded_image_path"])
            notifications = fetch_notifications_for_user(
                connection,
                user_id=int(user["user_id"]),
                role=ROLE_FINDER,
                limit=25,
            )
            create_activity_log(
                connection=connection,
                action="view_finder_dashboard",
                entity_type="users",
                entity_id=int(user["user_id"]),
                user_id=int(user["user_id"]),
                details={"report_count": len(reports)},
            )
        return {"reports": reports, "notifications": notifications}
    except sqlite3.Error as exc:
        logger.exception("Database error while loading finder dashboard user_id=%s", user.get("user_id"))
        raise RoleDashboardError("Unable to load child finder dashboard.") from exc


def get_authority_dashboard(user: dict[str, Any]) -> dict[str, Any]:
    _require_role(user, ROLE_AUTHORITY)
    try:
        overview = get_dashboard_overview()
        initialize_database()
        with database_transaction() as connection:
            reports = fetch_found_child_reports(connection, limit=50)
            for report in reports:
                report["matches"] = fetch_report_matches(connection, int(report["report_id"]))
                report["absolute_image_path"] = _absolute_path(report["uploaded_image_path"])
            notifications = fetch_notifications_for_user(
                connection,
                user_id=int(user["user_id"]),
                role=ROLE_AUTHORITY,
                limit=25,
            )
            create_activity_log(
                connection=connection,
                action="view_authority_dashboard",
                entity_type="users",
                entity_id=int(user["user_id"]),
                user_id=int(user["user_id"]),
                details={"report_count": len(reports), "notification_count": len(notifications)},
            )
        return {"overview": overview, "found_reports": reports, "notifications": notifications}
    except sqlite3.Error as exc:
        logger.exception("Database error while loading authority dashboard user_id=%s", user.get("user_id"))
        raise RoleDashboardError("Unable to load authority dashboard.") from exc


def submit_found_child_report(
    user: dict[str, Any],
    uploaded_file: Any,
    location: str,
    description: str,
) -> dict[str, Any]:
    _require_role(user, ROLE_FINDER)
    normalized_location = _required_text(location, "Found location", 200)
    normalized_description = _optional_text(description, "Description", 500)

    try:
        match_result = search_found_child(uploaded_file)
    except MatchingDatabaseError as exc:
        raise RoleDashboardError(str(exc)) from exc

    matches = match_result.get("matches", [])
    status = FOUND_REPORT_STATUS_PENDING if matches else FOUND_REPORT_STATUS_NO_MATCH

    try:
        initialize_database()
        with database_transaction() as connection:
            report_id = create_found_child_report(
                connection,
                {
                    "reporter_user_id": int(user["user_id"]),
                    "search_id": match_result["search_id"],
                    "uploaded_image_path": match_result["uploaded_image_path"],
                    "location": normalized_location,
                    "description": normalized_description,
                    "matches_found": len(matches),
                    "best_similarity_score": float(match_result["best_similarity_score"]),
                    "status": status,
                },
            )
            create_found_child_report_matches(connection, report_id, matches)
            notification_count = _create_match_notifications(
                connection=connection,
                report_id=report_id,
                matches=matches,
                location=normalized_location,
            )
            create_activity_log(
                connection=connection,
                action="submit_found_child_report",
                entity_type="found_child_reports",
                entity_id=report_id,
                user_id=int(user["user_id"]),
                details={
                    "search_id": match_result["search_id"],
                    "matches_found": len(matches),
                    "best_similarity_score": float(match_result["best_similarity_score"]),
                    "notification_count": notification_count,
                },
            )

        logger.info(
            "Found child report submitted report_id=%s user_id=%s matches=%s",
            report_id,
            user["user_id"],
            len(matches),
        )
        return {
            "report_id": report_id,
            "status": status,
            "matches": matches,
            "matches_found": len(matches),
            "best_similarity_score": float(match_result["best_similarity_score"]),
            "search_id": match_result["search_id"],
            "notification_count": notification_count,
        }
    except sqlite3.Error as exc:
        logger.exception("Database error while saving found child report user_id=%s", user.get("user_id"))
        raise RoleDashboardError("Unable to save found child report.") from exc


def change_found_report_status(user: dict[str, Any], report_id: int, status: str) -> None:
    _require_role(user, ROLE_AUTHORITY)
    normalized_status = _choice(status, "Report status", FOUND_REPORT_STATUSES)
    try:
        initialize_database()
        with database_transaction() as connection:
            updated = update_found_child_report_status(connection, int(report_id), normalized_status)
            if not updated:
                raise ValidationError("Found-child report was not found.")
            create_activity_log(
                connection=connection,
                action="update_found_report_status",
                entity_type="found_child_reports",
                entity_id=int(report_id),
                user_id=int(user["user_id"]),
                details={"status": normalized_status},
            )
    except ValidationError:
        raise
    except sqlite3.Error as exc:
        logger.exception("Database error while updating found report status report_id=%s", report_id)
        raise RoleDashboardError("Unable to update found-child report status.") from exc


def mark_user_notification_read(user: dict[str, Any], notification_id: int) -> None:
    try:
        initialize_database()
        with database_transaction() as connection:
            mark_notification_read(connection, int(notification_id), int(user["user_id"]))
    except sqlite3.Error as exc:
        logger.exception("Database error while marking notification read notification_id=%s", notification_id)
        raise RoleDashboardError("Unable to update notification.") from exc


def calculate_missing_days(last_seen_date: Any) -> int:
    if not last_seen_date:
        return 0
    try:
        parsed = datetime.strptime(str(last_seen_date), "%Y-%m-%d").date()
    except ValueError:
        return 0
    return max((date.today() - parsed).days, 0)


def _create_match_notifications(
    connection,
    report_id: int,
    matches: list[dict[str, Any]],
    location: str,
) -> int:
    notification_count = 0
    for match in matches:
        confidence = float(match["similarity_score"]) * 100.0
        title = f"AI match detected for {match['child_name']}"
        message = (
            f"Case {match['case_id']} received a found-child report with {confidence:.2f}% "
            f"similarity. Location: {location}."
        )
        create_notification(
            connection,
            {
                "recipient_role": ROLE_AUTHORITY,
                "child_id": int(match["child_id"]),
                "report_id": report_id,
                "case_id": match["case_id"],
                "title": title,
                "message": message,
                "notification_type": "ai_match_detected",
            },
        )
        notification_count += 1

        parent_recipients = fetch_parent_user_recipients_for_child(connection, int(match["child_id"]))
        if parent_recipients:
            for recipient in parent_recipients:
                create_notification(
                    connection,
                    {
                        "recipient_user_id": int(recipient["user_id"]),
                        "recipient_role": ROLE_PARENT,
                        "child_id": int(match["child_id"]),
                        "report_id": report_id,
                        "case_id": match["case_id"],
                        "title": title,
                        "message": message,
                        "notification_type": "ai_match_detected",
                    },
                )
                notification_count += 1
        else:
            create_notification(
                connection,
                {
                    "recipient_role": ROLE_PARENT,
                    "child_id": int(match["child_id"]),
                    "report_id": report_id,
                    "case_id": match["case_id"],
                    "title": title,
                    "message": message,
                    "notification_type": "ai_match_detected",
                },
            )
            notification_count += 1
    return notification_count


def _require_role(user: dict[str, Any], role: str) -> None:
    if not user or user.get("role") != role:
        raise ValidationError("You do not have permission to access this page.")


def _required_text(value: Any, field_name: str, max_length: int) -> str:
    cleaned = " ".join(str(value or "").split())
    if not cleaned:
        raise ValidationError(f"{field_name} is required.")
    if len(cleaned) > max_length:
        raise ValidationError(f"{field_name} must be {max_length} characters or fewer.")
    return cleaned


def _optional_text(value: Any, field_name: str, max_length: int) -> str | None:
    cleaned = " ".join(str(value or "").split())
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        raise ValidationError(f"{field_name} must be {max_length} characters or fewer.")
    return cleaned


def _choice(value: Any, field_name: str, choices: list[str]) -> str:
    cleaned = _required_text(value, field_name, 80)
    if cleaned not in choices:
        raise ValidationError(f"{field_name} has an invalid value.")
    return cleaned


def _absolute_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)
