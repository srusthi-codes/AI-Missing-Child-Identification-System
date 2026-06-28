import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.model_assets import ModelAssetError, ModelAssetSpec, ensure_model_file
from config.settings import (
    FACE_EMBEDDING_BACKEND,
    MIN_EMBEDDING_L2_NORM,
    MIN_FACE_AREA_RATIO,
    MIN_FACE_IMAGE_QUALITY_SCORE,
    OPENCV_FACE_NMS_THRESHOLD,
    OPENCV_FACE_SCORE_THRESHOLD,
    OPENCV_FACE_TOP_K,
    OPENCV_SFACE_EMBEDDING_DIMENSION,
    OPENCV_SFACE_MODEL_SHA256,
    OPENCV_SFACE_MODEL_PATH,
    OPENCV_SFACE_MODEL_SIZE,
    OPENCV_SFACE_MODEL_URLS,
    OPENCV_YUNET_MODEL_SHA256,
    OPENCV_YUNET_MODEL_PATH,
    OPENCV_YUNET_MODEL_SIZE,
    OPENCV_YUNET_MODEL_URLS,
)
from utils.logger import get_logger


logger = get_logger(__name__)


class FaceEmbeddingError(Exception):
    """Base class for face embedding infrastructure errors."""


class FaceImageRejectedError(FaceEmbeddingError):
    """Raised when a single image is not suitable for embedding generation."""


class FaceEmbeddingDependencyError(FaceEmbeddingError):
    """Raised when embedding dependencies or model assets are unavailable."""


@dataclass(frozen=True)
class FaceEmbeddingResult:
    image_path: str
    image_hash: str
    embedding: list[float]
    model_name: str
    detector_backend: str
    embedding_dimension: int
    quality_score: float
    face_confidence: float | None


def generate_face_embedding_for_image(image_record: dict[str, Any]) -> FaceEmbeddingResult:
    image_path = Path(image_record["absolute_path"])

    if not image_path.exists() or not image_path.is_file():
        raise FaceImageRejectedError("Saved image file is missing.")

    quality_score = calculate_image_quality_score(image_path)
    if quality_score < MIN_FACE_IMAGE_QUALITY_SCORE:
        raise FaceImageRejectedError(
            f"Image quality score {quality_score:.1f} is below the required "
            f"{MIN_FACE_IMAGE_QUALITY_SCORE:.1f}."
        )

    if FACE_EMBEDDING_BACKEND == "opencv_sface":
        return _generate_opencv_sface_embedding(image_record, image_path, quality_score)

    raise FaceEmbeddingDependencyError(f"Unsupported face embedding backend: {FACE_EMBEDDING_BACKEND}")


def calculate_image_quality_score(image_path: Path) -> float:
    cv2, np = _load_quality_dependencies()

    image = cv2.imread(str(image_path))
    if image is None:
        raise FaceImageRejectedError("Image could not be read by OpenCV.")

    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise FaceImageRejectedError("Image dimensions are invalid.")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    brightness = float(np.mean(grayscale))

    sharpness_score = min(sharpness / 120.0, 1.0) * 60.0
    brightness_score = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5) * 25.0
    resolution_score = min((width * height) / float(240 * 240), 1.0) * 15.0

    return round(sharpness_score + brightness_score + resolution_score, 2)


def _generate_opencv_sface_embedding(
    image_record: dict[str, Any],
    image_path: Path,
    quality_score: float,
) -> FaceEmbeddingResult:
    cv2, _ = _load_quality_dependencies()
    _ensure_opencv_face_api(cv2)

    try:
        yunet_model_path = ensure_model_file(
            ModelAssetSpec(
                path=OPENCV_YUNET_MODEL_PATH,
                urls=OPENCV_YUNET_MODEL_URLS,
                sha256=OPENCV_YUNET_MODEL_SHA256,
                size_bytes=OPENCV_YUNET_MODEL_SIZE,
            )
        )
        sface_model_path = ensure_model_file(
            ModelAssetSpec(
                path=OPENCV_SFACE_MODEL_PATH,
                urls=OPENCV_SFACE_MODEL_URLS,
                sha256=OPENCV_SFACE_MODEL_SHA256,
                size_bytes=OPENCV_SFACE_MODEL_SIZE,
            )
        )
    except ModelAssetError as exc:
        raise FaceEmbeddingDependencyError(str(exc)) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise FaceImageRejectedError("Image could not be read by OpenCV.")

    height, width = image.shape[:2]

    try:
        detector = cv2.FaceDetectorYN_create(
            str(yunet_model_path),
            "",
            (width, height),
            OPENCV_FACE_SCORE_THRESHOLD,
            OPENCV_FACE_NMS_THRESHOLD,
            OPENCV_FACE_TOP_K,
        )
        _, faces = detector.detect(image)
    except Exception as exc:
        logger.exception("OpenCV YuNet failed while detecting faces in %s", image_path)
        raise FaceEmbeddingError("OpenCV YuNet face detection failed.") from exc

    face, face_area_ratio = _select_single_face(faces, image_width=width, image_height=height)

    try:
        recognizer = cv2.FaceRecognizerSF_create(str(sface_model_path), "")
        aligned_face = recognizer.alignCrop(image, face)
        raw_embedding = recognizer.feature(aligned_face)
    except Exception as exc:
        logger.exception("OpenCV SFace failed while generating an embedding for %s", image_path)
        raise FaceEmbeddingError("OpenCV SFace embedding generation failed.") from exc

    embedding = _normalize_embedding(
        raw_embedding.flatten().tolist(),
        expected_dimension=OPENCV_SFACE_EMBEDDING_DIMENSION,
    )

    if not embedding:
        raise FaceImageRejectedError("OpenCV SFace returned an empty embedding vector.")

    face_confidence = float(face[-1])
    logger.info(
        "Generated OpenCV SFace embedding image=%s dimension=%s quality_score=%.2f face_confidence=%.4f face_area_ratio=%.4f",
        image_record["image_path"],
        len(embedding),
        quality_score,
        face_confidence,
        face_area_ratio,
    )

    return FaceEmbeddingResult(
        image_path=image_record["image_path"],
        image_hash=image_record["image_hash"],
        embedding=embedding,
        model_name="OpenCV-SFace",
        detector_backend="OpenCV-YuNet",
        embedding_dimension=len(embedding),
        quality_score=quality_score,
        face_confidence=face_confidence,
    )


def _select_single_face(faces: Any, image_width: int, image_height: int) -> tuple[Any, float]:
    if faces is None or len(faces) == 0:
        raise FaceImageRejectedError("No face detected.")

    valid_faces = [face for face in faces if float(face[-1]) >= OPENCV_FACE_SCORE_THRESHOLD]
    if not valid_faces:
        raise FaceImageRejectedError("No face met the minimum detection confidence.")

    if len(valid_faces) > 1:
        raise FaceImageRejectedError("Multiple faces detected. Upload an image with only the child.")

    face = valid_faces[0]
    face_area_ratio = _calculate_face_area_ratio(face, image_width, image_height)
    if face_area_ratio < MIN_FACE_AREA_RATIO:
        raise FaceImageRejectedError(
            f"Detected face is too small in the image. Face area ratio {face_area_ratio:.4f} "
            f"is below the required {MIN_FACE_AREA_RATIO:.4f}."
        )

    return face, face_area_ratio


def _calculate_face_area_ratio(face: Any, image_width: int, image_height: int) -> float:
    face_width = max(float(face[2]), 0.0)
    face_height = max(float(face[3]), 0.0)
    image_area = float(image_width * image_height)
    if image_area <= 0:
        return 0.0
    return (face_width * face_height) / image_area


def _normalize_embedding(value: Any, expected_dimension: int | None = None) -> list[float]:
    if value is None:
        return []

    try:
        embedding = [float(number) for number in value]
    except (TypeError, ValueError) as exc:
        raise FaceEmbeddingError("Embedding provider returned a non-numeric vector.") from exc

    if any(not math.isfinite(number) for number in embedding):
        raise FaceEmbeddingError("Embedding vector contains non-finite values.")

    if expected_dimension is not None and len(embedding) != expected_dimension:
        raise FaceEmbeddingError(
            f"Embedding dimension {len(embedding)} does not match expected {expected_dimension}."
        )

    l2_norm = math.sqrt(sum(number * number for number in embedding))
    if l2_norm < MIN_EMBEDDING_L2_NORM:
        raise FaceEmbeddingError("Embedding vector norm is too small.")

    return embedding


def _load_quality_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise FaceEmbeddingDependencyError(
            "OpenCV and NumPy are required for face embedding and image quality checks."
        ) from exc
    return cv2, np


def _ensure_opencv_face_api(cv2: Any) -> None:
    if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
        raise FaceEmbeddingDependencyError(
            "Installed OpenCV does not include YuNet/SFace APIs. Install dependencies from requirements.txt."
        )
