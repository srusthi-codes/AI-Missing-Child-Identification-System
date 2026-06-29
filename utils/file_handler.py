import hashlib
import re
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from config.constants import ALLOWED_IMAGE_EXTENSIONS, ALLOWED_IMAGE_MIME_TYPES
from config.settings import (
    BASE_DIR,
    CHILD_IMAGE_DIR,
    FOUND_CHILD_UPLOAD_DIR,
    MAX_IMAGES_PER_CHILD,
    MAX_UPLOAD_SIZE_MB,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
)
from utils.logger import get_logger
from utils.validators import ValidationError


logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedImage:
    original_filename: str
    content_type: str
    file_size: int
    image_hash: str
    image: Image.Image


def validate_uploaded_images(uploaded_files: list[Any] | None) -> list[PreparedImage]:
    if not uploaded_files:
        raise ValidationError("Please upload at least one child image.")

    if len(uploaded_files) > MAX_IMAGES_PER_CHILD:
        raise ValidationError(f"Upload a maximum of {MAX_IMAGES_PER_CHILD} images per child.")

    prepared_images: list[PreparedImage] = []
    seen_hashes: set[str] = set()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    for uploaded_file in uploaded_files:
        original_filename = Path(getattr(uploaded_file, "name", "")).name
        content_type = (getattr(uploaded_file, "type", "") or "").lower()
        suffix = Path(original_filename).suffix.lower()

        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValidationError(f"{original_filename} is not a supported image type.")

        file_bytes = uploaded_file.getvalue()
        file_size = len(file_bytes)

        if file_size == 0:
            raise ValidationError(f"{original_filename} is empty.")

        if file_size > max_bytes:
            raise ValidationError(f"{original_filename} exceeds the {MAX_UPLOAD_SIZE_MB} MB file limit.")

        image_hash = hashlib.sha256(file_bytes).hexdigest()
        if image_hash in seen_hashes:
            raise ValidationError(f"{original_filename} is a duplicate of another selected image.")

        try:
            with Image.open(BytesIO(file_bytes)) as probe:
                image_format = (probe.format or "").upper()
                probe.verify()

            with Image.open(BytesIO(file_bytes)) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValidationError(f"{original_filename} is not a valid image file.") from exc

        if image_format not in {"JPEG", "PNG"}:
            raise ValidationError(f"{original_filename} must be a JPEG or PNG image.")

        if content_type and content_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise ValidationError(f"{original_filename} has an invalid image content type.")

        if image.width < MIN_IMAGE_WIDTH or image.height < MIN_IMAGE_HEIGHT:
            raise ValidationError(
                f"{original_filename} is too small. Minimum size is "
                f"{MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT} pixels."
            )

        prepared_images.append(
            PreparedImage(
                original_filename=original_filename,
                content_type=content_type or _content_type_from_format(image_format),
                file_size=file_size,
                image_hash=image_hash,
                image=image.copy(),
            )
        )
        seen_hashes.add(image_hash)

    return prepared_images


def save_uploaded_images(case_id: str, prepared_images: list[PreparedImage]) -> list[dict[str, Any]]:
    safe_case_id = _safe_path_segment(case_id)
    target_dir = CHILD_IMAGE_DIR / safe_case_id
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_records: list[dict[str, Any]] = []

    for prepared_image in prepared_images:
        image_filename = f"{uuid.uuid4().hex}.jpg"
        image_path = target_dir / image_filename

        prepared_image.image.save(image_path, format="JPEG", quality=90, optimize=True)
        relative_path = image_path.relative_to(BASE_DIR)

        saved_records.append(
            {
                "image_path": str(relative_path),
                "absolute_path": str(image_path),
                "original_filename": prepared_image.original_filename,
                "content_type": "image/jpeg",
                "file_size": image_path.stat().st_size,
                "image_hash": prepared_image.image_hash,
            }
        )

    logger.info("Saved %s registration images for case_id=%s", len(saved_records), case_id)
    return saved_records


def save_found_child_search_image(prepared_image: PreparedImage) -> dict[str, Any]:
    FOUND_CHILD_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    image_filename = f"{uuid.uuid4().hex}.jpg"
    image_path = FOUND_CHILD_UPLOAD_DIR / image_filename

    prepared_image.image.save(image_path, format="JPEG", quality=90, optimize=True)
    relative_path = image_path.relative_to(BASE_DIR)

    record = {
        "image_path": str(relative_path),
        "absolute_path": str(image_path),
        "original_filename": prepared_image.original_filename,
        "content_type": "image/jpeg",
        "file_size": image_path.stat().st_size,
        "image_hash": prepared_image.image_hash,
    }

    logger.info("Saved found child search image path=%s", relative_path)
    return record


def cleanup_saved_files(paths: list[str]) -> None:
    for file_path in paths:
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            logger.warning("Could not remove orphaned image file: %s", file_path)


def _content_type_from_format(image_format: str) -> str:
    if image_format == "PNG":
        return "image/png"
    return "image/jpeg"


def _safe_path_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", value.strip())
    if not cleaned:
        raise ValidationError("Invalid case ID for image storage.")
    return cleaned
