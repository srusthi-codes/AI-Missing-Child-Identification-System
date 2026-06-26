from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.model_assets import ModelAssetError, ensure_model_file
from config.settings import (
    DEEPFACE_DETECTOR_BACKEND,
    DEEPFACE_MODEL_NAME,
    FACE_EMBEDDING_BACKEND,
    MIN_FACE_IMAGE_QUALITY_SCORE,
    OPENCV_FACE_NMS_THRESHOLD,
    OPENCV_FACE_SCORE_THRESHOLD,
    OPENCV_FACE_TOP_K,
    OPENCV_SFACE_MODEL_PATH,
    OPENCV_SFACE_MODEL_URL,
    OPENCV_YUNET_MODEL_PATH,
    OPENCV_YUNET_MODEL_URL,
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

    if FACE_EMBEDDING_BACKEND == "deepface":
        return _generate_deepface_embedding(image_record, image_path, quality_score)

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
        yunet_model_path = ensure_model_file(OPENCV_YUNET_MODEL_PATH, OPENCV_YUNET_MODEL_URL)
        sface_model_path = ensure_model_file(OPENCV_SFACE_MODEL_PATH, OPENCV_SFACE_MODEL_URL)
    except ModelAssetError as exc:
        raise FaceEmbeddingDependencyError(str(exc)) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise FaceImageRejectedError("Image could not be read by OpenCV.")

    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN_create(
        str(yunet_model_path),
        "",
        (width, height),
        OPENCV_FACE_SCORE_THRESHOLD,
        OPENCV_FACE_NMS_THRESHOLD,
        OPENCV_FACE_TOP_K,
    )

    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        raise FaceImageRejectedError("No face detected.")

    valid_faces = [face for face in faces if float(face[-1]) >= OPENCV_FACE_SCORE_THRESHOLD]
    if not valid_faces:
        raise FaceImageRejectedError("No face met the minimum detection confidence.")

    if len(valid_faces) > 1:
        raise FaceImageRejectedError("Multiple faces detected. Upload an image with only the child.")

    face = valid_faces[0]
    recognizer = cv2.FaceRecognizerSF_create(str(sface_model_path), "")
    aligned_face = recognizer.alignCrop(image, face)
    raw_embedding = recognizer.feature(aligned_face)
    embedding = _normalize_embedding(raw_embedding.flatten().tolist())

    if not embedding:
        raise FaceImageRejectedError("OpenCV SFace returned an empty embedding vector.")

    return FaceEmbeddingResult(
        image_path=image_record["image_path"],
        image_hash=image_record["image_hash"],
        embedding=embedding,
        model_name="OpenCV-SFace",
        detector_backend="OpenCV-YuNet",
        embedding_dimension=len(embedding),
        quality_score=quality_score,
        face_confidence=float(face[-1]),
    )


def _generate_deepface_embedding(
    image_record: dict[str, Any],
    image_path: Path,
    quality_score: float,
) -> FaceEmbeddingResult:
    face_representations = _represent_faces_with_deepface(image_path)

    if not face_representations:
        raise FaceImageRejectedError("No face detected.")

    if len(face_representations) > 1:
        raise FaceImageRejectedError("Multiple faces detected. Upload an image with only the child.")

    representation = face_representations[0]
    embedding = _normalize_embedding(representation.get("embedding"))

    if not embedding:
        raise FaceImageRejectedError("DeepFace returned an empty embedding vector.")

    return FaceEmbeddingResult(
        image_path=image_record["image_path"],
        image_hash=image_record["image_hash"],
        embedding=embedding,
        model_name=DEEPFACE_MODEL_NAME,
        detector_backend=DEEPFACE_DETECTOR_BACKEND,
        embedding_dimension=len(embedding),
        quality_score=quality_score,
        face_confidence=_normalize_optional_float(representation.get("face_confidence")),
    )


def _represent_faces_with_deepface(image_path: Path) -> list[dict[str, Any]]:
    DeepFace = _load_deepface()

    try:
        result = DeepFace.represent(
            img_path=str(image_path),
            model_name=DEEPFACE_MODEL_NAME,
            detector_backend=DEEPFACE_DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
    except ValueError as exc:
        message = str(exc).lower()
        if "face could not be detected" in message or "no face" in message:
            raise FaceImageRejectedError("No face detected.") from exc
        raise FaceImageRejectedError(f"DeepFace rejected the image: {exc}") from exc
    except Exception as exc:
        logger.exception("DeepFace failed while generating an embedding for %s", image_path)
        raise FaceEmbeddingError("DeepFace failed while generating a face embedding.") from exc

    if isinstance(result, dict):
        return [result]

    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]

    raise FaceEmbeddingError("DeepFace returned an unsupported embedding response.")


def _normalize_embedding(value: Any) -> list[float]:
    if value is None:
        return []

    try:
        return [float(number) for number in value]
    except (TypeError, ValueError) as exc:
        raise FaceEmbeddingError("Embedding provider returned a non-numeric vector.") from exc


def _normalize_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_deepface() -> Any:
    try:
        from deepface import DeepFace
    except (ImportError, ModuleNotFoundError) as exc:
        raise FaceEmbeddingDependencyError(
            "DeepFace is not importable in this environment. Use FACE_EMBEDDING_BACKEND='opencv_sface'."
        ) from exc
    return DeepFace


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
