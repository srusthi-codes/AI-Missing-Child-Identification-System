import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from ai.embedding_generator import (
    FaceEmbeddingDependencyError,
    FaceEmbeddingError,
    FaceImageRejectedError,
    generate_face_embedding_for_image,
)
from config.settings import BASE_DIR, MATCH_SIMILARITY_THRESHOLD, MATCH_TOP_K, OPENCV_SFACE_EMBEDDING_DIMENSION
from database.connection import database_transaction
from database.repositories.embedding_repository import fetch_all_embedding_candidates
from database.repositories.log_repository import create_activity_log
from database.repositories.search_history_repository import (
    count_searches_by_image_hash,
    create_search_history_record,
)
from database.schema import initialize_database
from utils.file_handler import save_found_child_search_image, validate_uploaded_images
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


class MatchingDatabaseError(Exception):
    """Raised when matching cannot complete due to database failure."""


class InvalidStoredEmbeddingError(Exception):
    """Raised when a stored embedding row cannot be used for matching."""


def search_found_child(uploaded_file: Any) -> dict[str, Any]:
    prepared_image = _validate_single_search_image(uploaded_file)
    saved_image = save_found_child_search_image(prepared_image)

    try:
        query_embedding = generate_face_embedding_for_image(saved_image).embedding
    except FaceImageRejectedError as exc:
        _record_failed_search(saved_image, str(exc))
        logger.warning("Found child search image rejected: %s", exc)
        raise ValidationError(str(exc)) from exc
    except FaceEmbeddingDependencyError as exc:
        _record_failed_search(saved_image, str(exc))
        logger.exception("Found child search failed because embedding dependencies are unavailable")
        raise ValidationError(
            "Face embedding service is unavailable. Install project dependencies and ensure model access."
        ) from exc
    except FaceEmbeddingError as exc:
        _record_failed_search(saved_image, str(exc))
        logger.exception("Found child search embedding generation failed")
        raise ValidationError("Face embedding generation failed. Please upload a clearer image.") from exc

    try:
        initialize_database()
        with database_transaction() as connection:
            duplicate_search_count = count_searches_by_image_hash(connection, saved_image["image_hash"])
            candidates = fetch_all_embedding_candidates(connection)
            ranked_matches, invalid_embedding_count = rank_embedding_matches(query_embedding, candidates)
            threshold_matches = [
                match
                for match in ranked_matches
                if match["similarity_score"] >= MATCH_SIMILARITY_THRESHOLD
            ]
            top_matches = threshold_matches[:MATCH_TOP_K]
            best_similarity_score = ranked_matches[0]["similarity_score"] if ranked_matches else 0.0

            search_id = create_search_history_record(
                connection,
                {
                    "uploaded_image_path": saved_image["image_path"],
                    "image_hash": saved_image["image_hash"],
                    "matches_found": len(top_matches),
                    "best_similarity_score": best_similarity_score,
                    "status": "completed",
                    "error_message": None,
                },
            )

            create_activity_log(
                connection=connection,
                action="found_child_search",
                entity_type="search_history",
                entity_id=search_id,
                details={
                    "uploaded_image_path": saved_image["image_path"],
                    "candidate_count": len(candidates),
                    "invalid_embedding_count": invalid_embedding_count,
                    "matches_found": len(top_matches),
                    "best_similarity_score": best_similarity_score,
                    "duplicate_search_count": duplicate_search_count,
                },
            )

        logger.info(
            "Found child search completed search_id=%s matches_found=%s best_similarity=%.4f candidates=%s invalid_embeddings=%s",
            search_id,
            len(top_matches),
            best_similarity_score,
            len(candidates),
            invalid_embedding_count,
        )

        return {
            "search_id": search_id,
            "uploaded_image_path": saved_image["image_path"],
            "uploaded_image_absolute_path": saved_image["absolute_path"],
            "matches": top_matches,
            "matches_found": len(top_matches),
            "best_similarity_score": best_similarity_score,
            "threshold": MATCH_SIMILARITY_THRESHOLD,
            "top_k": MATCH_TOP_K,
            "candidate_count": len(candidates),
            "invalid_embedding_count": invalid_embedding_count,
            "duplicate_search_count": duplicate_search_count,
        }

    except sqlite3.Error as exc:
        logger.exception("Database error while running found child search")
        raise MatchingDatabaseError("Unable to complete search due to a database error.") from exc


def rank_embedding_matches(
    query_embedding: list[float],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    normalized_query = normalize_embedding_vector(query_embedding)
    matches: list[dict[str, Any]] = []
    invalid_embedding_count = 0

    for candidate in candidates:
        try:
            candidate_embedding = _deserialize_candidate_embedding(candidate)
            normalized_candidate = normalize_embedding_vector(candidate_embedding)
        except InvalidStoredEmbeddingError as exc:
            invalid_embedding_count += 1
            logger.warning(
                "Skipping invalid stored embedding embedding_id=%s reason=%s",
                candidate.get("embedding_id"),
                exc,
            )
            continue

        similarity_score = cosine_similarity_score(normalized_query, normalized_candidate)
        matches.append(_candidate_to_match(candidate, similarity_score))

    matches.sort(key=lambda item: item["similarity_score"], reverse=True)
    return matches, invalid_embedding_count


def normalize_embedding_vector(embedding: list[float]) -> list[float]:
    try:
        vector = [float(value) for value in embedding]
    except (TypeError, ValueError) as exc:
        raise InvalidStoredEmbeddingError("Embedding vector contains non-numeric values.") from exc

    if len(vector) != OPENCV_SFACE_EMBEDDING_DIMENSION:
        raise InvalidStoredEmbeddingError(
            f"Embedding dimension {len(vector)} does not match expected {OPENCV_SFACE_EMBEDDING_DIMENSION}."
        )

    if any(not math.isfinite(value) for value in vector):
        raise InvalidStoredEmbeddingError("Embedding vector contains non-finite values.")

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        raise InvalidStoredEmbeddingError("Embedding vector has zero norm.")

    return [value / norm for value in vector]


def cosine_similarity_score(normalized_embedding_a: list[float], normalized_embedding_b: list[float]) -> float:
    cosine_value = sum(a * b for a, b in zip(normalized_embedding_a, normalized_embedding_b))
    cosine_value = max(-1.0, min(1.0, cosine_value))
    return round(max(0.0, min(1.0, (cosine_value + 1.0) / 2.0)), 6)


def _validate_single_search_image(uploaded_file: Any) -> Any:
    if uploaded_file is None:
        raise ValidationError("Please upload a found child image.")

    prepared_images = validate_uploaded_images([uploaded_file])
    if len(prepared_images) != 1:
        raise ValidationError("Upload exactly one found child image.")

    return prepared_images[0]


def _record_failed_search(saved_image: dict[str, Any], error_message: str) -> None:
    try:
        initialize_database()
        with database_transaction() as connection:
            search_id = create_search_history_record(
                connection,
                {
                    "uploaded_image_path": saved_image["image_path"],
                    "image_hash": saved_image["image_hash"],
                    "matches_found": 0,
                    "best_similarity_score": 0.0,
                    "status": "failed",
                    "error_message": error_message[:500],
                },
            )
            create_activity_log(
                connection=connection,
                action="found_child_search_failed",
                entity_type="search_history",
                entity_id=search_id,
                details={"error_message": error_message[:500]},
            )
    except sqlite3.Error:
        logger.exception("Could not store failed found child search history")


def _deserialize_candidate_embedding(candidate: dict[str, Any]) -> list[float]:
    raw_embedding = candidate.get("embedding")
    try:
        embedding = json.loads(raw_embedding) if isinstance(raw_embedding, str) else raw_embedding
    except json.JSONDecodeError as exc:
        raise InvalidStoredEmbeddingError("Stored embedding is not valid JSON.") from exc

    if not isinstance(embedding, list):
        raise InvalidStoredEmbeddingError("Stored embedding is not a list.")

    return embedding


def _candidate_to_match(candidate: dict[str, Any], similarity_score: float) -> dict[str, Any]:
    stored_image_path = _absolute_path(candidate["image_path"])
    return {
        "embedding_id": candidate["embedding_id"],
        "child_id": candidate["child_id"],
        "case_id": candidate["case_id"],
        "child_name": candidate["full_name"],
        "age": candidate["age"],
        "gender": candidate["gender"],
        "guardian_name": candidate.get("guardian_name") or "Not provided",
        "contact_number": candidate.get("guardian_phone") or "Not provided",
        "stored_image_path": str(stored_image_path),
        "stored_image_relative_path": candidate["image_path"],
        "similarity_score": similarity_score,
        "similarity_percentage": round(similarity_score * 100.0, 2),
    }


def _absolute_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path
