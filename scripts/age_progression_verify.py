import base64
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.age_progression_generator import (
    AgeProgressionError,
    analyze_face_image,
    ensure_age_progression_ready,
    generate_age_progressed_estimate,
)
from ai.embedding_generator import FaceImageRejectedError, generate_face_embedding_for_image
from config.constants import MAX_CHILD_AGE
from config.settings import BASE_DIR, TEMP_DIR
from database.connection import database_transaction
from database.repositories.embedding_repository import fetch_all_embedding_candidates
from database.schema import initialize_database
from services.age_progression_service import (
    get_age_progression_case_details,
    get_age_progression_children,
    get_age_progression_history,
    generate_age_progression_preview,
    remove_age_progression_result_for_verification,
    save_age_progression_result,
)
from services.dashboard_service import get_dashboard_overview
from services.matching_service import rank_embedding_matches
from services.records_service import get_registered_children
from utils.validators import ValidationError


VERIFY_DIR = TEMP_DIR / "age_progression_verify"


def main() -> None:
    created_history_ids: list[int] = []
    created_files: list[Path] = []

    try:
        initialize_database()
        ensure_age_progression_ready()
        print("OK: age progression model readiness passed.")

        sample = _find_existing_sample()
        print(f"Using existing case: {sample['case_id']} child_id={sample['child_id']}")

        _verify_invalid_target_ages(sample)
        _verify_invalid_source_images()
        _verify_dynamic_progression_strength(sample)

        preview = generate_age_progression_preview(
            child_id=sample["child_id"],
            source_image_id=sample["image_id"],
            target_age=sample["target_age"],
        )
        assert preview["identity_score"] > 0
        assert preview["generated_image_b64"]
        print("OK: successful age progression preview generated.")

        saved = save_age_progression_result(preview)
        created_history_ids.append(saved["history_id"])
        generated_path = _absolute_path(saved["generated_image_path"])
        created_files.append(generated_path)
        assert generated_path.exists()
        print(f"OK: generated estimate saved at {saved['generated_image_path']}.")

        generated_analysis = analyze_face_image(generated_path)
        assert generated_analysis.face_confidence > 0
        generated_embedding = generate_face_embedding_for_image(
            {
                "absolute_path": str(generated_path),
                "image_path": saved["generated_image_path"],
                "image_hash": generated_analysis.image_hash,
            }
        )
        assert generated_embedding.embedding_dimension == 128
        print("OK: generated image is readable, face-detectable, and SFace-compatible.")

        history = get_age_progression_history(sample["child_id"])
        assert any(item["history_id"] == saved["history_id"] for item in history)
        print("OK: age progression database persistence and history retrieval passed.")

        _verify_existing_modules(sample)
        print("OK: existing registration, records, matching, embedding, and dashboard modules remain callable.")

    finally:
        for history_id in created_history_ids:
            remove_age_progression_result_for_verification(history_id)
        for path in created_files:
            if path.exists():
                path.unlink()
        if VERIFY_DIR.exists():
            shutil.rmtree(VERIFY_DIR)

    print("OK: Phase Age Progression verification passed.")


def _find_existing_sample() -> dict[str, object]:
    children = get_age_progression_children()
    if not children:
        raise AssertionError("No existing child records are available for age progression verification.")

    fallback_sample: dict[str, object] | None = None
    for child in children:
        source_age = int(child["age"])
        if source_age >= MAX_CHILD_AGE:
            continue
        details = get_age_progression_case_details(int(child["child_id"]))
        for image in details.get("images", []):
            image_path = _absolute_path(image["image_path"])
            if image_path.exists():
                sample = {
                    "child_id": int(child["child_id"]),
                    "case_id": child["case_id"],
                    "source_age": source_age,
                    "target_age": min(source_age + 7, MAX_CHILD_AGE),
                    "image_id": int(image["image_id"]),
                    "image_path": image["image_path"],
                }
                if source_age <= MAX_CHILD_AGE - 29:
                    return sample
                if fallback_sample is None:
                    fallback_sample = sample

    if fallback_sample is not None:
        return fallback_sample
    raise AssertionError("No existing child record with an available source image was found.")


def _verify_invalid_target_ages(sample: dict[str, object]) -> None:
    source_age = int(sample["source_age"])
    for invalid_age in (source_age, max(source_age - 1, 0)):
        try:
            generate_age_progression_preview(
                child_id=int(sample["child_id"]),
                source_image_id=int(sample["image_id"]),
                target_age=invalid_age,
            )
        except ValidationError:
            continue
        raise AssertionError(f"Invalid target age {invalid_age} was not rejected.")
    print("OK: invalid target ages are rejected.")


def _verify_invalid_source_images() -> None:
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    no_face_path = VERIFY_DIR / "no_face.jpg"
    no_face_image = Image.new("RGB", (240, 240), color=(245, 245, 245))
    draw = ImageDraw.Draw(no_face_image)
    for offset in range(0, 240, 16):
        draw.line((0, offset, 239, 239 - offset), fill=(30, 90, 110), width=2)
        draw.line((offset, 0, 239 - offset, 239), fill=(90, 150, 160), width=2)
    no_face_image.save(no_face_path)

    corrupt_path = VERIFY_DIR / "corrupt.jpg"
    corrupt_path.write_bytes(b"not a valid image")

    unsupported_path = VERIFY_DIR / "unsupported.txt"
    unsupported_path.write_text("not an image", encoding="utf-8")

    missing_path = VERIFY_DIR / "missing.jpg"

    invalid_paths = [
        (no_face_path, "No face"),
        (missing_path, "missing image"),
        (corrupt_path, "corrupted image"),
        (unsupported_path, "unsupported image"),
    ]
    for path, label in invalid_paths:
        try:
            generate_age_progressed_estimate(path, source_age=8, target_age=15)
        except (FaceImageRejectedError, AgeProgressionError) as exc:
            message = str(exc)
            if label == "No face" and "No face" not in message:
                raise AssertionError(f"No-face validation returned an unexpected message: {message}") from exc
            if label == "missing image" and "missing" not in message.lower():
                raise AssertionError(f"Missing-image validation returned an unexpected message: {message}") from exc
            continue
        raise AssertionError(f"{label} source image was not rejected.")
    print("OK: no-face, missing, corrupted, and unsupported source images are rejected.")


def _verify_dynamic_progression_strength(sample: dict[str, object]) -> None:
    source_age = int(sample["source_age"])
    target_ages = _dynamic_target_ages(source_age)
    if len(target_ages) < 2:
        raise AssertionError("Selected sample does not have enough valid future target ages for progression testing.")

    source_image_path = _absolute_path(str(sample["image_path"]))
    source_image = cv2.imread(str(source_image_path))
    if source_image is None:
        raise AssertionError("Verification source image could not be read.")

    deltas: list[float] = []
    identity_scores: list[float] = []
    for target_age in target_ages:
        preview = generate_age_progression_preview(
            child_id=int(sample["child_id"]),
            source_image_id=int(sample["image_id"]),
            target_age=target_age,
        )
        generated_bytes = base64.b64decode(preview["generated_image_b64"].encode("ascii"))
        generated_image = cv2.imdecode(np.frombuffer(generated_bytes, np.uint8), cv2.IMREAD_COLOR)
        if generated_image is None:
            raise AssertionError(f"Generated image for target age {target_age} could not be decoded.")
        if generated_image.shape[:2] != source_image.shape[:2]:
            generated_image = cv2.resize(generated_image, (source_image.shape[1], source_image.shape[0]))

        face_delta = _mean_face_region_delta(source_image, generated_image, preview["source_analysis"]["bbox"])
        deltas.append(face_delta)
        identity_scores.append(float(preview["identity_score"]))

        generated_analysis = preview["generated_analysis"]
        if float(generated_analysis["face_confidence"]) <= 0:
            raise AssertionError(f"Generated face was not detectable for target age {target_age}.")
        if not preview["target_age_label"]:
            raise AssertionError(f"Target age label missing for target age {target_age}.")

    if deltas[0] <= 0.35:
        raise AssertionError(f"Small-gap progression was visually identical to the source: delta={deltas[0]:.3f}.")
    if deltas[-1] <= deltas[0] + 3.0:
        raise AssertionError(
            "Large-gap progression did not become meaningfully stronger "
            f"(small={deltas[0]:.3f}, large={deltas[-1]:.3f})."
        )
    if any(next_delta + 0.75 < current_delta for current_delta, next_delta in zip(deltas, deltas[1:])):
        raise AssertionError(f"Progression deltas are not stage-consistent: {deltas}.")
    if min(identity_scores) <= 0:
        raise AssertionError(f"Identity-preservation scoring failed: {identity_scores}.")

    metrics = ", ".join(
        f"target {target}: delta={delta:.2f}, identity={identity:.3f}"
        for target, delta, identity in zip(target_ages, deltas, identity_scores)
    )
    print(f"OK: dynamic target-age progression strength validated ({metrics}).")


def _dynamic_target_ages(source_age: int) -> list[int]:
    remaining_years = MAX_CHILD_AGE - source_age
    if remaining_years <= 0:
        return []

    small_gap = min(2, remaining_years)
    medium_gap = min(max(6, remaining_years // 3), 14, remaining_years)
    large_gap = min(max(10, (remaining_years * 2) // 3), 29, remaining_years)
    return sorted({source_age + gap for gap in (small_gap, medium_gap, large_gap) if gap > 0})


def _mean_face_region_delta(source_image, generated_image, bbox: list[int] | tuple[int, int, int, int]) -> float:
    x, y, width, height = [int(value) for value in bbox]
    pad_x = int(width * 0.45)
    pad_y = int(height * 0.45)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(source_image.shape[1], x + width + pad_x)
    y1 = min(source_image.shape[0], y + height + pad_y)
    source_region = source_image[y0:y1, x0:x1].astype(np.float32)
    generated_region = generated_image[y0:y1, x0:x1].astype(np.float32)
    return float(np.mean(np.abs(generated_region - source_region)))


def _verify_existing_modules(sample: dict[str, object]) -> None:
    records = get_registered_children()
    assert isinstance(records, list)

    overview = get_dashboard_overview()
    assert overview["statistics"]["total_registered_children"] >= 0

    source_image_path = _absolute_path(str(sample["image_path"]))
    source_analysis = analyze_face_image(source_image_path)
    source_embedding = generate_face_embedding_for_image(
        {
            "absolute_path": str(source_image_path),
            "image_path": str(sample["image_path"]),
            "image_hash": source_analysis.image_hash,
        }
    )
    assert source_embedding.embedding_dimension == 128

    with database_transaction() as connection:
        candidates = fetch_all_embedding_candidates(connection)
    matches, _ = rank_embedding_matches(source_embedding.embedding, candidates[:10])
    assert isinstance(matches, list)


def _absolute_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


if __name__ == "__main__":
    main()
