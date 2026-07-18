import base64
import sqlite3
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai.age_progression_generator import (
    AgeProgressionDependencyError,
    AgeProgressionError,
    age_stage_label,
    ensure_age_progression_ready,
    generate_age_progressed_estimate,
)
from ai.embedding_generator import (
    FaceEmbeddingDependencyError,
    FaceEmbeddingError,
    FaceImageRejectedError,
    generate_face_embedding_for_image,
)
from config.constants import MAX_CHILD_AGE, MIN_CHILD_AGE
from config.settings import (
    AGE_PROGRESSION_DIR,
    AGE_PROGRESSION_MIN_IDENTITY_THRESHOLD,
    AGE_PROGRESSION_MODERATE_IDENTITY_THRESHOLD,
    AGE_PROGRESSION_STRONG_IDENTITY_THRESHOLD,
    BASE_DIR,
)
from database.connection import database_transaction
from database.repositories.age_progression_repository import (
    create_age_progression_record,
    delete_age_progression_record,
    fetch_age_progression_history,
)
from database.repositories.dashboard_repository import fetch_case_details, fetch_filtered_child_records
from database.repositories.log_repository import create_activity_log
from database.schema import initialize_database
from services.matching_service import cosine_similarity_score, normalize_embedding_vector
from utils.file_handler import cleanup_saved_files
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)

DISCLAIMER = (
    "AI-generated age-progressed estimate. This image represents a possible appearance only and must not be "
    "treated as an exact prediction, confirmed appearance, or confirmed identity."
)


class AgeProgressionDatabaseError(Exception):
    """Raised when age progression data cannot be loaded or saved."""


def get_age_progression_children(search_value: str = "") -> list[dict[str, Any]]:
    normalized_search = " ".join(str(search_value or "").split()).lower()
    try:
        initialize_database()
        with database_transaction() as connection:
            records = fetch_filtered_child_records(connection, {})

        if not normalized_search:
            return records

        return [
            record
            for record in records
            if normalized_search in str(record["case_id"]).lower()
            or normalized_search in str(record["full_name"]).lower()
            or normalized_search in str(record["status"]).lower()
        ]
    except sqlite3.Error as exc:
        logger.exception("Database error while loading age progression child choices")
        raise AgeProgressionDatabaseError("Unable to load child records for age progression.") from exc


def get_age_progression_case_details(child_id: int) -> dict[str, Any]:
    normalized_child_id = _validate_child_id(child_id)
    try:
        initialize_database()
        with database_transaction() as connection:
            case_details = fetch_case_details(connection, normalized_child_id)
            if case_details is None:
                raise ValidationError("Selected child case was not found.")
            history = fetch_age_progression_history(connection, normalized_child_id)
            create_activity_log(
                connection=connection,
                action="view_age_progression_case",
                entity_type="missing_child",
                entity_id=normalized_child_id,
                details={"case_id": case_details["case_id"], "history_count": len(history)},
            )

        case_details["age_progression_history"] = history
        return case_details
    except ValidationError:
        raise
    except sqlite3.Error as exc:
        logger.exception("Database error while loading age progression case details child_id=%s", normalized_child_id)
        raise AgeProgressionDatabaseError("Unable to load age progression case details.") from exc


def generate_age_progression_preview(
    child_id: int,
    source_image_id: int,
    target_age: int,
) -> dict[str, Any]:
    case_details = get_age_progression_case_details(child_id)
    source_image = _select_source_image(case_details, source_image_id)
    source_age = int(case_details["age"])
    normalized_target_age = _validate_target_age(source_age, target_age)
    source_absolute_path = _absolute_path(source_image["image_path"])

    try:
        ensure_age_progression_ready()
        render_result = generate_age_progressed_estimate(
            source_image_path=source_absolute_path,
            source_age=source_age,
            target_age=normalized_target_age,
        )
        identity_score, identity_quality = _score_identity_preservation(
            source_image_path=source_absolute_path,
            generated_image_bytes=render_result.image_bytes,
            source_image_relative_path=source_image["image_path"],
            source_image_hash=source_image["image_hash"],
        )
    except (AgeProgressionDependencyError, FaceEmbeddingDependencyError) as exc:
        logger.exception("Age progression dependencies are unavailable")
        raise ValidationError("Age progression service is unavailable. Check model files and dependencies.") from exc
    except FaceImageRejectedError as exc:
        logger.info("Age progression source image rejected child_id=%s reason=%s", child_id, exc)
        raise ValidationError(str(exc)) from exc
    except (AgeProgressionError, FaceEmbeddingError) as exc:
        logger.exception("Age progression generation failed child_id=%s case_id=%s", child_id, case_details["case_id"])
        raise ValidationError(f"Age progression generation failed: {exc}") from exc

    preview = {
        "child_id": case_details["child_id"],
        "case_id": case_details["case_id"],
        "child_name": case_details["full_name"],
        "source_image_id": source_image_id,
        "source_image_path": source_image["image_path"],
        "source_image_absolute_path": str(source_absolute_path),
        "source_age": source_age,
        "target_age": normalized_target_age,
        "target_age_label": render_result.target_age_label,
        "progression_years": render_result.progression_years,
        "model_name": render_result.model_name,
        "approach_notes": render_result.approach_notes,
        "identity_score": identity_score,
        "identity_quality": identity_quality,
        "disclaimer": DISCLAIMER,
        "generated_image_b64": base64.b64encode(render_result.image_bytes).decode("ascii"),
        "source_analysis": asdict(render_result.source_analysis),
        "generated_analysis": asdict(render_result.generated_analysis),
    }

    logger.info(
        "Age progression preview generated child_id=%s case_id=%s source_age=%s target_age=%s identity_score=%.4f",
        child_id,
        case_details["case_id"],
        source_age,
        normalized_target_age,
        identity_score,
    )
    return preview


def save_age_progression_result(preview: dict[str, Any]) -> dict[str, Any]:
    required_keys = {
        "child_id",
        "case_id",
        "source_image_path",
        "source_age",
        "target_age",
        "target_age_label",
        "progression_years",
        "model_name",
        "identity_score",
        "identity_quality",
        "generated_image_b64",
    }
    missing_keys = sorted(key for key in required_keys if key not in preview)
    if missing_keys:
        raise ValidationError("Age progression preview is incomplete. Generate a new estimate before saving.")

    child_id = _validate_child_id(preview["child_id"])
    generated_bytes = _decode_generated_image(preview["generated_image_b64"])
    saved_path = _save_generated_image_bytes(preview["case_id"], generated_bytes)

    try:
        initialize_database()
        with database_transaction() as connection:
            case_details = fetch_case_details(connection, child_id)
            if case_details is None:
                raise ValidationError("Selected child case was not found.")
            if case_details["case_id"] != preview["case_id"]:
                raise ValidationError("Age progression preview no longer matches the selected case.")

            history_id = create_age_progression_record(
                connection,
                {
                    "child_id": child_id,
                    "case_id": preview["case_id"],
                    "source_image_path": preview["source_image_path"],
                    "generated_image_path": str(saved_path.relative_to(BASE_DIR)),
                    "source_age": int(preview["source_age"]),
                    "target_age": int(preview["target_age"]),
                    "target_age_label": preview["target_age_label"],
                    "progression_years": int(preview["progression_years"]),
                    "model_name": preview["model_name"],
                    "identity_score": float(preview["identity_score"]),
                    "identity_quality": preview["identity_quality"],
                },
            )
            create_activity_log(
                connection=connection,
                action="save_age_progression_result",
                entity_type="age_progression_history",
                entity_id=history_id,
                details={
                    "child_id": child_id,
                    "case_id": preview["case_id"],
                    "source_age": int(preview["source_age"]),
                    "target_age": int(preview["target_age"]),
                    "identity_score": float(preview["identity_score"]),
                    "identity_quality": preview["identity_quality"],
                },
            )

        logger.info(
            "Saved age progression result history_id=%s child_id=%s case_id=%s path=%s",
            history_id,
            child_id,
            preview["case_id"],
            saved_path,
        )
        return {
            "history_id": history_id,
            "generated_image_path": str(saved_path.relative_to(BASE_DIR)),
            "absolute_generated_image_path": str(saved_path),
        }
    except ValidationError:
        cleanup_saved_files([str(saved_path)])
        raise
    except sqlite3.Error as exc:
        cleanup_saved_files([str(saved_path)])
        logger.exception("Database error while saving age progression result child_id=%s", child_id)
        raise AgeProgressionDatabaseError("Unable to save age progression result.") from exc


def get_age_progression_history(child_id: int) -> list[dict[str, Any]]:
    normalized_child_id = _validate_child_id(child_id)
    try:
        initialize_database()
        with database_transaction() as connection:
            return fetch_age_progression_history(connection, normalized_child_id)
    except sqlite3.Error as exc:
        logger.exception("Database error while retrieving age progression history child_id=%s", normalized_child_id)
        raise AgeProgressionDatabaseError("Unable to retrieve age progression history.") from exc


def remove_age_progression_result_for_verification(history_id: int) -> dict[str, Any] | None:
    try:
        initialize_database()
        with database_transaction() as connection:
            deleted = delete_age_progression_record(connection, int(history_id))
        if deleted and deleted.get("generated_image_path"):
            cleanup_saved_files([str(_absolute_path(deleted["generated_image_path"]))])
        return deleted
    except sqlite3.Error as exc:
        logger.exception("Database error while removing verification age progression result")
        raise AgeProgressionDatabaseError("Unable to clean up verification age progression result.") from exc


def identity_quality_label(score: float) -> str:
    if score >= AGE_PROGRESSION_STRONG_IDENTITY_THRESHOLD:
        return "Strong identity preservation"
    if score >= AGE_PROGRESSION_MODERATE_IDENTITY_THRESHOLD:
        return "Moderate identity preservation"
    return "Low identity preservation"


def _score_identity_preservation(
    source_image_path: Path,
    generated_image_bytes: bytes,
    source_image_relative_path: str,
    source_image_hash: str,
) -> tuple[float, str]:
    temporary_path = _write_temporary_validation_image(generated_image_bytes)
    try:
        source_record = {
            "absolute_path": str(source_image_path),
            "image_path": source_image_relative_path,
            "image_hash": source_image_hash,
        }
        generated_record = {
            "absolute_path": str(temporary_path),
            "image_path": str(temporary_path),
            "image_hash": _sha256_bytes(generated_image_bytes),
        }
        source_embedding = generate_face_embedding_for_image(source_record).embedding
        generated_embedding = generate_face_embedding_for_image(generated_record).embedding
        normalized_source = normalize_embedding_vector(source_embedding)
        normalized_generated = normalize_embedding_vector(generated_embedding)
        score = cosine_similarity_score(normalized_source, normalized_generated)
        return score, identity_quality_label(score)
    finally:
        cleanup_saved_files([str(temporary_path)])


def _write_temporary_validation_image(image_bytes: bytes) -> Path:
    target_dir = BASE_DIR / "storage" / "temp" / "age_progression_validation"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{uuid.uuid4().hex}.jpg"
    target_path.write_bytes(image_bytes)
    return target_path


def _save_generated_image_bytes(case_id: str, image_bytes: bytes) -> Path:
    safe_case_id = _safe_path_segment(case_id)
    target_dir = AGE_PROGRESSION_DIR / safe_case_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{uuid.uuid4().hex}.jpg"
    target_path.write_bytes(image_bytes)
    return target_path


def _decode_generated_image(value: str) -> bytes:
    try:
        image_bytes = base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValidationError("Generated age progression image data is invalid.") from exc
    if not image_bytes:
        raise ValidationError("Generated age progression image data is empty.")
    return image_bytes


def _select_source_image(case_details: dict[str, Any], source_image_id: int) -> dict[str, Any]:
    try:
        normalized_image_id = int(source_image_id)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Select a valid source image.") from exc

    for image in case_details.get("images", []):
        if int(image["image_id"]) == normalized_image_id:
            absolute_path = _absolute_path(image["image_path"])
            if not absolute_path.exists() or not absolute_path.is_file():
                raise ValidationError("Selected source image file is missing.")
            return image
    raise ValidationError("Selected source image does not belong to this child record.")


def _validate_target_age(source_age: int, target_age: Any) -> int:
    try:
        normalized_target_age = int(target_age)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Target age must be a valid number.") from exc

    if normalized_target_age < MIN_CHILD_AGE or normalized_target_age > MAX_CHILD_AGE:
        raise ValidationError(f"Target age must be between {MIN_CHILD_AGE} and {MAX_CHILD_AGE}.")
    if normalized_target_age <= source_age:
        raise ValidationError("Target age must be greater than the registered age.")
    return normalized_target_age


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


def _safe_path_segment(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "_" for character in str(value))
    if not cleaned:
        raise ValidationError("Invalid case ID for age progression storage.")
    return cleaned


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def target_age_label(age: int) -> str:
    return age_stage_label(age)


def is_identity_preservation_low(score: float) -> bool:
    return score < AGE_PROGRESSION_MIN_IDENTITY_THRESHOLD
