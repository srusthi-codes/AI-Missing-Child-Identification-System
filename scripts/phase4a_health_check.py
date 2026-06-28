import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.embedding_generator import FaceEmbeddingDependencyError  # noqa: E402
from ai.model_assets import ModelAssetSpec, ensure_model_file  # noqa: E402
from config.settings import (  # noqa: E402
    DATABASE_PATH,
    FACE_EMBEDDING_BACKEND,
    OPENCV_SFACE_MODEL_SHA256,
    OPENCV_SFACE_MODEL_PATH,
    OPENCV_SFACE_MODEL_SIZE,
    OPENCV_SFACE_MODEL_URLS,
    OPENCV_FACE_NMS_THRESHOLD,
    OPENCV_FACE_SCORE_THRESHOLD,
    OPENCV_FACE_TOP_K,
    OPENCV_YUNET_MODEL_SHA256,
    OPENCV_YUNET_MODEL_PATH,
    OPENCV_YUNET_MODEL_SIZE,
    OPENCV_YUNET_MODEL_URLS,
)
from database.schema import initialize_database  # noqa: E402


def main() -> int:
    print("Phase 4A health check")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Embedding backend: {FACE_EMBEDDING_BACKEND}")

    try:
        _check_python_packages()
        _check_opencv_face_apis()
        _check_database_schema()
        _check_model_assets()
        _check_opencv_model_loading()
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    print("OK: Phase 4A embedding infrastructure is ready.")
    return 0


def _check_python_packages() -> None:
    import cv2
    import numpy as np
    import PIL

    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Pillow: {PIL.__version__}")


def _check_opencv_face_apis() -> None:
    import cv2

    missing_apis = [
        api_name
        for api_name in ("FaceDetectorYN_create", "FaceRecognizerSF_create")
        if not hasattr(cv2, api_name)
    ]
    if missing_apis:
        raise FaceEmbeddingDependencyError(
            "OpenCV is missing required face APIs: " + ", ".join(missing_apis)
        )
    print("OK: OpenCV YuNet/SFace APIs are available.")


def _check_database_schema() -> None:
    initialize_database()

    connection = sqlite3.connect(DATABASE_PATH)
    try:
        row = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'face_embeddings'
            """
        ).fetchone()
        if row is None:
            raise RuntimeError("face_embeddings table was not created.")
    finally:
        connection.close()

    print(f"OK: SQLite schema is ready at {DATABASE_PATH}")


def _check_model_assets() -> None:
    ensure_model_file(
        ModelAssetSpec(
            path=OPENCV_YUNET_MODEL_PATH,
            urls=OPENCV_YUNET_MODEL_URLS,
            sha256=OPENCV_YUNET_MODEL_SHA256,
            size_bytes=OPENCV_YUNET_MODEL_SIZE,
        )
    )
    ensure_model_file(
        ModelAssetSpec(
            path=OPENCV_SFACE_MODEL_PATH,
            urls=OPENCV_SFACE_MODEL_URLS,
            sha256=OPENCV_SFACE_MODEL_SHA256,
            size_bytes=OPENCV_SFACE_MODEL_SIZE,
        )
    )

    print(f"OK: YuNet model ready at {OPENCV_YUNET_MODEL_PATH}")
    print(f"OK: SFace model ready at {OPENCV_SFACE_MODEL_PATH}")


def _check_opencv_model_loading() -> None:
    import cv2

    cv2.FaceDetectorYN_create(
        str(OPENCV_YUNET_MODEL_PATH),
        "",
        (320, 320),
        OPENCV_FACE_SCORE_THRESHOLD,
        OPENCV_FACE_NMS_THRESHOLD,
        OPENCV_FACE_TOP_K,
    )
    cv2.FaceRecognizerSF_create(str(OPENCV_SFACE_MODEL_PATH), "")

    print("OK: OpenCV can parse and load YuNet/SFace ONNX models.")


if __name__ == "__main__":
    raise SystemExit(main())
