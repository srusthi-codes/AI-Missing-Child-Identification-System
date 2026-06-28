import argparse
import json
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATABASE_PATH, OPENCV_SFACE_EMBEDDING_DIMENSION, TEMP_DIR  # noqa: E402
from services.registration_service import register_missing_child  # noqa: E402
from utils.validators import ValidationError  # noqa: E402


@dataclass(frozen=True)
class LocalUploadedFile:
    path: Path
    type: str

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


@dataclass(frozen=True)
class BytesUploadedFile:
    name: str
    type: str
    payload: bytes

    def getvalue(self) -> bytes:
        return self.payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Register one child with one or more local images and verify that Phase 4B "
            "stores valid OpenCV SFace embeddings linked to the same child record."
        )
    )
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Path to a clear single-face JPG or PNG image. Repeat this option for multiple images.",
    )
    parser.add_argument(
        "--skip-duplicate-test",
        action="store_true",
        help="Skip the duplicate-upload validation check.",
    )
    args = parser.parse_args()

    source_image_paths = [_resolve_image_path(value) for value in args.image]
    verification_image_paths = _materialize_unique_verification_images(source_image_paths)
    uploaded_files = [
        LocalUploadedFile(path=image_path, type=_content_type_for_path(image_path))
        for image_path in verification_image_paths
    ]

    try:
        result = register_missing_child(_child_data(), _parent_data(), uploaded_files)
        verification = _verify_registration_rows(result["child_id"])

        if not args.skip_duplicate_test:
            _verify_existing_image_rejection(verification_image_paths[0])
            _verify_duplicate_upload_rejection(verification_image_paths[0])
            _verify_invalid_upload_rejections()

    except Exception as exc:
        print(f"FAILED: Phase 4B verification did not complete: {exc}")
        return 1

    print("OK: Phase 4B registration and embedding verification passed.")
    print(f"Case ID: {result['case_id']}")
    print(f"Child ID: {result['child_id']}")
    print(f"Uploaded images: {verification['image_count']}")
    print(f"Stored embeddings: {verification['embedding_count']}")
    print(f"Embedding dimensions: {verification['embedding_dimensions']}")
    return 0


def _resolve_image_path(value: str) -> Path:
    image_path = Path(value).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        raise ValidationError(f"Image file does not exist: {image_path}")
    return image_path


def _materialize_unique_verification_images(image_paths: list[Path]) -> list[Path]:
    run_token = uuid.uuid4().hex
    target_dir = TEMP_DIR / "phase4b_verification_inputs" / run_token
    target_dir.mkdir(parents=True, exist_ok=True)
    color_seed = int(run_token[:6], 16)

    materialized_paths: list[Path] = []
    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        draw = ImageDraw.Draw(image)
        marker_color = (
            (color_seed + index * 31) % 255,
            (color_seed + index * 17) % 255,
            (color_seed + index * 7) % 255,
        )
        draw.rectangle((index, index, index + 2, index + 2), fill=marker_color)

        target_path = target_dir / f"phase4b_input_{index}.jpg"
        image.save(target_path, format="JPEG", quality=92)
        materialized_paths.append(target_path)

    return materialized_paths


def _verify_registration_rows(child_id: int) -> dict[str, object]:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        image_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM child_images WHERE child_id = ?",
                (child_id,),
            ).fetchone()[0]
        )
        embedding_rows = connection.execute(
            """
            SELECT embedding, embedding_dimension, model_name, detector_backend
            FROM face_embeddings
            WHERE child_id = ?
            """,
            (child_id,),
        ).fetchall()
    finally:
        connection.close()

    if image_count < 1:
        raise RuntimeError("No child image metadata rows were stored.")

    if not embedding_rows:
        raise RuntimeError("No face embedding rows were stored.")

    dimensions: list[int] = []
    for row in embedding_rows:
        dimension = int(row["embedding_dimension"])
        dimensions.append(dimension)

        if row["model_name"] != "OpenCV-SFace" or row["detector_backend"] != "OpenCV-YuNet":
            raise RuntimeError("Unexpected embedding model metadata was stored.")

        if dimension != OPENCV_SFACE_EMBEDDING_DIMENSION:
            raise RuntimeError(f"Unexpected embedding dimension {dimension}.")

        embedding = json.loads(row["embedding"])
        if not isinstance(embedding, list) or len(embedding) != OPENCV_SFACE_EMBEDDING_DIMENSION:
            raise RuntimeError("Stored embedding vector is malformed.")

    return {
        "image_count": image_count,
        "embedding_count": len(embedding_rows),
        "embedding_dimensions": sorted(set(dimensions)),
    }


def _verify_duplicate_upload_rejection(image_path: Path) -> None:
    duplicate_file_1 = LocalUploadedFile(path=image_path, type=_content_type_for_path(image_path))
    duplicate_file_2 = LocalUploadedFile(path=image_path, type=_content_type_for_path(image_path))

    try:
        register_missing_child(_child_data("Duplicate Upload Sample"), _parent_data("+911234567891"), [duplicate_file_1, duplicate_file_2])
    except ValidationError:
        print("OK: duplicate upload validation rejected repeated image in the same batch.")
        return

    raise RuntimeError("Duplicate upload validation did not reject repeated image.")


def _verify_existing_image_rejection(image_path: Path) -> None:
    uploaded_file = LocalUploadedFile(path=image_path, type=_content_type_for_path(image_path))

    try:
        register_missing_child(_child_data("Existing Image Duplicate Sample"), _parent_data("+911234567892"), [uploaded_file])
    except ValidationError:
        print("OK: database duplicate validation rejected an already registered image.")
        return

    raise RuntimeError("Database duplicate validation did not reject an already registered image.")


def _verify_invalid_upload_rejections() -> None:
    invalid_cases = [
        BytesUploadedFile(name="unsupported.gif", type="image/gif", payload=b"not an allowed format"),
        BytesUploadedFile(name="corrupted.jpg", type="image/jpeg", payload=b"not a real jpeg image"),
    ]

    for invalid_file in invalid_cases:
        try:
            register_missing_child(
                _child_data(f"Invalid Upload Sample {invalid_file.name}"),
                _parent_data("+911234567893"),
                [invalid_file],
            )
        except ValidationError:
            print(f"OK: invalid upload rejected: {invalid_file.name}")
            continue

        raise RuntimeError(f"Invalid upload was not rejected: {invalid_file.name}")


def _child_data(full_name: str = "Phase 4B Sample Child") -> dict[str, object]:
    return {
        "full_name": full_name,
        "age": 10,
        "gender": "Other",
        "identification_marks": "Phase 4B embedding verification sample",
        "last_seen_location": "Phase 4B Verification",
        "last_seen_date": date.today(),
        "last_seen_time": None,
        "description": "Temporary sample registration used to verify Phase 4B embedding storage.",
    }


def _parent_data(phone: str = "+911234567890") -> dict[str, object]:
    return {
        "guardian_name": "Phase 4B Verifier",
        "relationship": "Guardian",
        "phone": phone,
        "email": "phase4b.verify@example.com",
        "address": "Phase 4B Verification Address",
        "government_id_type": "",
        "government_id_last4": "",
    }


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    return "image/jpeg"


if __name__ == "__main__":
    raise SystemExit(main())
