from dataclasses import dataclass
from typing import Any

from ai.embedding_generator import (
    FaceEmbeddingDependencyError,
    FaceEmbeddingError,
    FaceEmbeddingResult,
    FaceImageRejectedError,
    generate_face_embedding_for_image,
)
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


@dataclass(frozen=True)
class RegistrationEmbeddingBatch:
    records: list[dict[str, Any]]
    rejected_reasons: list[str]


def generate_registration_embeddings(saved_images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return generate_registration_embedding_batch(saved_images).records


def generate_registration_embedding_batch(saved_images: list[dict[str, Any]]) -> RegistrationEmbeddingBatch:
    embedding_results: list[FaceEmbeddingResult] = []
    rejected_images: list[str] = []

    for image_record in saved_images:
        original_filename = image_record.get("original_filename", "uploaded image")

        try:
            embedding_results.append(generate_face_embedding_for_image(image_record))
        except FaceImageRejectedError as exc:
            reason = f"{original_filename}: {exc}"
            rejected_images.append(reason)
            logger.warning("Registration image rejected for embedding: %s", reason)
        except FaceEmbeddingDependencyError:
            raise
        except FaceEmbeddingError as exc:
            reason = f"{original_filename}: {exc}"
            rejected_images.append(reason)
            logger.exception("Embedding generation failed for image %s", original_filename)
            raise

    if not embedding_results:
        details = " ".join(rejected_images[:3])
        message = "No usable face embeddings could be generated from the uploaded images."
        if details:
            message = f"{message} {details}"
        raise ValidationError(message)

    logger.info(
        "Generated %s face embedding(s); rejected_image_count=%s",
        len(embedding_results),
        len(rejected_images),
    )

    records = [_embedding_result_to_record(result) for result in embedding_results]
    return RegistrationEmbeddingBatch(records=records, rejected_reasons=rejected_images)


def _embedding_result_to_record(result: FaceEmbeddingResult) -> dict[str, Any]:
    return {
        "image_path": result.image_path,
        "image_hash": result.image_hash,
        "embedding": result.embedding,
        "model_name": result.model_name,
        "detector_backend": result.detector_backend,
        "embedding_dimension": result.embedding_dimension,
        "quality_score": result.quality_score,
        "face_confidence": result.face_confidence,
    }
