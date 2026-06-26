import secrets
import sqlite3
from datetime import datetime
from typing import Any

from ai.embedding_generator import FaceEmbeddingDependencyError, FaceEmbeddingError
from config.constants import CHILD_STATUS_MISSING
from database.connection import database_transaction
from database.repositories.child_repository import case_id_exists, create_missing_child
from database.repositories.embedding_repository import create_face_embedding_records
from database.repositories.image_repository import create_child_image_records
from database.repositories.log_repository import create_activity_log
from database.repositories.parent_repository import create_parent_details
from database.schema import initialize_database
from services.embedding_service import generate_registration_embeddings
from utils.file_handler import cleanup_saved_files, save_uploaded_images, validate_uploaded_images
from utils.logger import get_logger
from utils.validators import ValidationError, validate_child_data, validate_parent_data


logger = get_logger(__name__)


def register_missing_child(
    child_data: dict[str, Any],
    parent_data: dict[str, Any],
    uploaded_files: list[Any] | None,
) -> dict[str, Any]:
    initialize_database()

    normalized_child = validate_child_data(child_data)
    normalized_parent = validate_parent_data(parent_data)
    prepared_images = validate_uploaded_images(uploaded_files)

    saved_image_paths: list[str] = []
    case_id = ""
    embedding_count = 0

    try:
        with database_transaction() as connection:
            case_id = _generate_case_id(connection)
            normalized_child["case_id"] = case_id
            normalized_child["status"] = CHILD_STATUS_MISSING
            normalized_child["registered_by"] = None

            saved_images = save_uploaded_images(case_id, prepared_images)
            saved_image_paths = [record["absolute_path"] for record in saved_images]
            embedding_records = generate_registration_embeddings(saved_images)
            embedding_count = len(embedding_records)

            child_id = create_missing_child(connection, normalized_child)
            parent_id = create_parent_details(connection, child_id, normalized_parent)
            create_child_image_records(connection, child_id, case_id, saved_images)
            create_face_embedding_records(connection, child_id, embedding_records)

            create_activity_log(
                connection=connection,
                action="register_missing_child",
                entity_type="missing_child",
                entity_id=child_id,
                details={
                    "case_id": case_id,
                    "child_name": normalized_child["full_name"],
                    "guardian_phone": normalized_parent["phone"],
                    "image_count": len(saved_images),
                    "embedding_count": embedding_count,
                    "embedding_model": embedding_records[0]["model_name"],
                },
            )

        logger.info(
            "Registered missing child case_id=%s image_count=%s embedding_count=%s",
            case_id,
            len(saved_image_paths),
            embedding_count,
        )
        return {
            "success": True,
            "case_id": case_id,
            "child_id": child_id,
            "parent_id": parent_id,
            "image_count": len(saved_image_paths),
            "embedding_count": embedding_count,
        }

    except ValidationError:
        cleanup_saved_files(saved_image_paths)
        raise
    except FaceEmbeddingDependencyError as exc:
        cleanup_saved_files(saved_image_paths)
        logger.exception("Face embedding dependencies are unavailable")
        raise ValidationError(
            "Face embedding service is unavailable. Install project dependencies and ensure model download access."
        ) from exc
    except FaceEmbeddingError as exc:
        cleanup_saved_files(saved_image_paths)
        logger.exception("Face embedding generation failed")
        raise ValidationError("Face embedding generation failed. Please upload clearer child images.") from exc
    except sqlite3.IntegrityError as exc:
        cleanup_saved_files(saved_image_paths)
        logger.warning("Registration failed due to integrity error: %s", exc)
        raise ValidationError(
            "This registration could not be saved because one of the uploaded images already exists."
        ) from exc
    except Exception:
        cleanup_saved_files(saved_image_paths)
        logger.exception("Registration failed for case_id=%s", case_id or "unassigned")
        raise


def _generate_case_id(connection: sqlite3.Connection) -> str:
    date_part = datetime.utcnow().strftime("%Y%m%d")
    for _ in range(10):
        random_part = secrets.token_hex(3).upper()
        case_id = f"MC-{date_part}-{random_part}"
        if not case_id_exists(connection, case_id):
            return case_id
    raise RuntimeError("Unable to generate a unique case ID")
