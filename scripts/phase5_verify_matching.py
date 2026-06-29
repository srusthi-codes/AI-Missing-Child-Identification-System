import argparse
import hashlib
import json
import math
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    DATABASE_PATH,
    MATCH_SIMILARITY_THRESHOLD,
    MATCH_TOP_K,
    OPENCV_SFACE_EMBEDDING_DIMENSION,
    TEMP_DIR,
)
from database.repositories.embedding_repository import fetch_all_embedding_candidates  # noqa: E402
from database.schema import SCHEMA_STATEMENTS, initialize_database  # noqa: E402
from services.matching_service import (  # noqa: E402
    cosine_similarity_score,
    normalize_embedding_vector,
    rank_embedding_matches,
    search_found_child,
)
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
        description="Verify Phase 5 found child matching with OpenCV YuNet/SFace embeddings."
    )
    parser.add_argument(
        "--image",
        help="Optional clear single-face JPG/PNG image. If omitted, an existing registered child image is reused.",
    )
    args = parser.parse_args()

    try:
        initialize_database()
        source_image = _resolve_source_image(args.image)
        work_dir = _make_work_dir()

        print(f"Using source image: {source_image}")
        print(f"Verification workspace: {work_dir}")

        _verify_empty_database()
        _verify_similarity_generation()
        _verify_top5_ranking()
        _verify_invalid_stored_embeddings()

        primary_child, query_path = _register_primary_phase5_child(source_image, work_dir)
        _verify_single_registered_child_match(primary_child, query_path)
        registered_children = [primary_child, *_register_additional_phase5_children(source_image, work_dir)]
        _verify_multiple_registered_children_match(registered_children)
        _verify_duplicate_search_image(query_path)
        _verify_no_match_scenario()

        _verify_no_face_image(work_dir)
        _verify_multiple_face_image(source_image, work_dir)
        _verify_corrupted_image()
        _verify_unsupported_image()

    except Exception as exc:
        print(f"FAILED: Phase 5 verification did not complete: {exc}")
        return 1

    print("OK: Phase 5 matching verification passed.")
    return 0


def _resolve_source_image(value: str | None) -> Path:
    if value:
        image_path = Path(value).expanduser().resolve()
        if not image_path.exists() or not image_path.is_file():
            raise RuntimeError(f"Image file does not exist: {image_path}")
        return image_path

    candidates = sorted(
        [
            path
            for extension in ("*.jpg", "*.jpeg", "*.png")
            for path in (PROJECT_ROOT / "storage" / "child_images").rglob(extension)
        ]
    )
    if not candidates:
        raise RuntimeError("No existing child image found. Register one child before running Phase 5 verification.")
    return candidates[0]


def _make_work_dir() -> Path:
    work_dir = TEMP_DIR / "phase5_verification_inputs" / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _verify_empty_database() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        candidates = fetch_all_embedding_candidates(connection)
    finally:
        connection.close()

    if candidates:
        raise RuntimeError("Empty database verification expected zero embedding candidates.")

    ranked_matches, invalid_count = rank_embedding_matches(_unit_vector(1.0), [])
    if ranked_matches or invalid_count != 0:
        raise RuntimeError("Empty embedding ranking should return no matches and no invalid rows.")

    print("OK: empty database and no-embedding scenarios returned no matches.")


def _verify_similarity_generation() -> None:
    normalized_a = normalize_embedding_vector(_unit_vector(1.0))
    normalized_b = normalize_embedding_vector(_unit_vector(0.5))
    score = cosine_similarity_score(normalized_a, normalized_b)

    if not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Similarity score is outside 0..1: {score}")

    expected = 0.75
    if abs(score - expected) > 0.000001:
        raise RuntimeError(f"Unexpected cosine similarity score {score}; expected {expected}.")

    print("OK: cosine similarity score generation is normalized to 0..1.")


def _verify_top5_ranking() -> None:
    candidates = [
        _synthetic_candidate(index=1, cosine_value=1.0),
        _synthetic_candidate(index=2, cosine_value=0.9),
        _synthetic_candidate(index=3, cosine_value=0.7),
        _synthetic_candidate(index=4, cosine_value=0.5),
        _synthetic_candidate(index=5, cosine_value=0.2),
        _synthetic_candidate(index=6, cosine_value=-1.0),
    ]

    ranked_matches, invalid_count = rank_embedding_matches(_unit_vector(1.0), candidates)
    top_five = ranked_matches[:MATCH_TOP_K]

    if invalid_count != 0:
        raise RuntimeError("Synthetic ranking should not contain invalid embeddings.")
    if len(top_five) != MATCH_TOP_K:
        raise RuntimeError(f"Expected {MATCH_TOP_K} ranked matches, got {len(top_five)}.")

    scores = [match["similarity_score"] for match in top_five]
    if scores != sorted(scores, reverse=True):
        raise RuntimeError("Top-5 results are not sorted by similarity.")
    if top_five[0]["case_id"] != "SYNTH-1" or top_five[-1]["case_id"] != "SYNTH-5":
        raise RuntimeError("Top-5 ranking order is incorrect.")

    print("OK: Top-5 ranking orders matches by cosine similarity.")


def _verify_invalid_stored_embeddings() -> None:
    candidates = [
        _synthetic_candidate(index=1, cosine_value=1.0),
        {**_synthetic_candidate(index=2, cosine_value=0.0), "embedding": "not-json"},
        {**_synthetic_candidate(index=3, cosine_value=0.0), "embedding": json.dumps([1.0, 0.0])},
    ]

    ranked_matches, invalid_count = rank_embedding_matches(_unit_vector(1.0), candidates)
    if invalid_count != 2:
        raise RuntimeError(f"Expected two invalid stored embeddings, got {invalid_count}.")
    if len(ranked_matches) != 1 or ranked_matches[0]["case_id"] != "SYNTH-1":
        raise RuntimeError("Invalid stored embeddings were not skipped cleanly.")

    print("OK: invalid stored embeddings are skipped with a valid result preserved.")


def _register_primary_phase5_child(source_image: Path, work_dir: Path) -> tuple[dict[str, Any], Path]:
    first_child_images = [
        _materialize_unique_image(source_image, work_dir / "child_1_image_1.jpg", marker_index=1),
        _materialize_unique_image(source_image, work_dir / "child_1_image_2.jpg", marker_index=2),
    ]
    first_child = _register_child(index=1, image_paths=first_child_images)
    first_child["image_paths"] = first_child_images

    query_path = _materialize_unique_image(source_image, work_dir / "query_single.jpg", marker_index=99)
    if not query_path.exists():
        raise RuntimeError("Could not materialize Phase 5 query image.")

    if first_child.get("embedding_count", 0) < 2:
        raise RuntimeError("Multiple image registration did not store multiple embeddings for the same child.")

    print("OK: registered primary Phase 5 child with multiple OpenCV SFace embeddings.")
    return first_child, query_path


def _register_additional_phase5_children(source_image: Path, work_dir: Path) -> list[dict[str, Any]]:
    registered_children: list[dict[str, Any]] = []

    for index in range(2, 7):
        image_path = _materialize_unique_image(
            source_image,
            work_dir / f"child_{index}_image_1.jpg",
            marker_index=index + 10,
        )
        child = _register_child(index=index, image_paths=[image_path])
        child["image_paths"] = [image_path]
        registered_children.append(child)

    print("OK: registered additional Phase 5 children with OpenCV SFace embeddings.")
    return registered_children


def _verify_single_registered_child_match(expected_child: dict[str, Any], query_path: Path) -> None:
    result = search_found_child(LocalUploadedFile(path=query_path, type="image/jpeg"))
    _verify_search_history_row(result)
    _assert_valid_search_result(result)

    expected_child_id = expected_child["child_id"]
    matched_child_ids = {match["child_id"] for match in result["matches"]}
    if expected_child_id not in matched_child_ids:
        raise RuntimeError("Single registered child was not returned in the Top-5 matches.")

    print("OK: single registered child search returned the expected child in Top-5.")


def _verify_multiple_registered_children_match(registered_children: list[dict[str, Any]]) -> None:
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        placeholders = ", ".join("?" for _ in registered_children)
        child_ids = [child["child_id"] for child in registered_children]
        row = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM face_embeddings
            WHERE child_id IN ({placeholders})
            """,
            child_ids,
        ).fetchone()
    finally:
        connection.close()

    embedding_count = int(row[0] if row else 0)
    if embedding_count < len(registered_children):
        raise RuntimeError("Multiple registered children do not have matching embedding rows.")

    print("OK: multiple registered children have embeddings available for matching.")


def _verify_duplicate_search_image(query_path: Path) -> None:
    first_result = search_found_child(LocalUploadedFile(path=query_path, type="image/jpeg"))
    second_result = search_found_child(LocalUploadedFile(path=query_path, type="image/jpeg"))

    if second_result["duplicate_search_count"] <= first_result["duplicate_search_count"]:
        raise RuntimeError("Duplicate search image was not detected through search history.")

    print("OK: duplicate search image is tracked through search history.")


def _verify_no_match_scenario() -> None:
    opposite_candidate = _synthetic_candidate(index=1, cosine_value=-1.0)
    ranked_matches, invalid_count = rank_embedding_matches(_unit_vector(1.0), [opposite_candidate])
    threshold_matches = [
        match for match in ranked_matches if match["similarity_score"] >= MATCH_SIMILARITY_THRESHOLD
    ]

    if invalid_count != 0:
        raise RuntimeError("No-match scenario should not produce invalid embeddings.")
    if threshold_matches:
        raise RuntimeError("No-match scenario unexpectedly crossed the configured threshold.")

    print("OK: no-match scenario returns no threshold-qualified matches.")


def _verify_no_face_image(work_dir: Path) -> None:
    no_face_path = _create_no_face_image(work_dir / "no_face.jpg")
    _expect_validation_error(
        LocalUploadedFile(path=no_face_path, type="image/jpeg"),
        "no-face image",
    )
    print("OK: no-face image is rejected.")


def _verify_multiple_face_image(source_image: Path, work_dir: Path) -> None:
    multi_face_path = _create_multiple_face_image(source_image, work_dir / "multiple_faces.jpg")
    _expect_validation_error(
        LocalUploadedFile(path=multi_face_path, type="image/jpeg"),
        "multiple-face image",
    )
    print("OK: multiple-face image is rejected.")


def _verify_corrupted_image() -> None:
    _expect_validation_error(
        BytesUploadedFile(name="corrupted.jpg", type="image/jpeg", payload=b"this is not a valid image"),
        "corrupted image",
    )
    print("OK: corrupted image is rejected.")


def _verify_unsupported_image() -> None:
    _expect_validation_error(
        BytesUploadedFile(name="unsupported.gif", type="image/gif", payload=b"GIF89a"),
        "unsupported image",
    )
    print("OK: unsupported image format is rejected.")


def _register_child(index: int, image_paths: list[Path]) -> dict[str, Any]:
    uploaded_files = [
        LocalUploadedFile(path=image_path, type=_content_type_for_path(image_path))
        for image_path in image_paths
    ]
    return register_missing_child(
        _child_data(index),
        _parent_data(index),
        uploaded_files,
    )


def _verify_search_history_row(result: dict[str, Any]) -> None:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT uploaded_image_path, matches_found, best_similarity_score, status
            FROM search_history
            WHERE id = ?
            """,
            (result["search_id"],),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError("Search history row was not stored.")
    if row["status"] != "completed":
        raise RuntimeError("Search history row was not marked completed.")
    if int(row["matches_found"]) != result["matches_found"]:
        raise RuntimeError("Search history match count does not match service result.")
    if abs(float(row["best_similarity_score"]) - float(result["best_similarity_score"])) > 0.000001:
        raise RuntimeError("Search history best similarity does not match service result.")


def _assert_valid_search_result(result: dict[str, Any]) -> None:
    if result["candidate_count"] <= 0:
        raise RuntimeError("Search did not evaluate any stored embeddings.")
    if result["matches_found"] <= 0:
        raise RuntimeError("Expected at least one threshold-qualified match.")
    if len(result["matches"]) > MATCH_TOP_K:
        raise RuntimeError(f"Search returned more than Top-{MATCH_TOP_K} matches.")

    previous_score = 1.0
    for match in result["matches"]:
        score = float(match["similarity_score"])
        if not 0.0 <= score <= 1.0:
            raise RuntimeError(f"Match similarity score is outside 0..1: {score}")
        if score < MATCH_SIMILARITY_THRESHOLD:
            raise RuntimeError("Search returned a match below the configured threshold.")
        if score > previous_score:
            raise RuntimeError("Search results are not sorted by similarity.")
        previous_score = score


def _expect_validation_error(uploaded_file: Any, label: str) -> None:
    try:
        search_found_child(uploaded_file)
    except ValidationError:
        return
    raise RuntimeError(f"Expected ValidationError for {label}.")


def _materialize_unique_image(source_image: Path, target_path: Path, marker_index: int) -> Path:
    with Image.open(source_image) as source:
        image = source.convert("RGB")

    marker_seed = int(hashlib.sha256(str(target_path).encode("utf-8")).hexdigest()[:8], 16)
    draw = ImageDraw.Draw(image)
    marker_color = (
        (marker_seed + marker_index * 53) % 255,
        (marker_seed + marker_index * 97) % 255,
        (marker_seed + marker_index * 193) % 255,
    )
    marker_x = marker_seed % 24
    marker_y = (marker_seed // 24) % 24
    draw.rectangle((marker_x, marker_y, marker_x + 4, marker_y + 4), fill=marker_color)
    image.save(target_path, format="JPEG", quality=92)
    return target_path


def _create_no_face_image(target_path: Path) -> Path:
    size = 512
    block_size = 16
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for y in range(0, size, block_size):
        for x in range(0, size, block_size):
            if (x // block_size + y // block_size) % 2 == 0:
                draw.rectangle((x, y, x + block_size - 1, y + block_size - 1), fill=(30, 30, 30))
    image.save(target_path, format="JPEG", quality=95)
    return target_path


def _create_multiple_face_image(source_image: Path, target_path: Path) -> Path:
    with Image.open(source_image) as source:
        face_image = source.convert("RGB")

    max_panel_width = 420
    if face_image.width > max_panel_width:
        scale = max_panel_width / float(face_image.width)
        face_image = face_image.resize((max_panel_width, max(1, int(face_image.height * scale))))

    padding = 40
    canvas = Image.new(
        "RGB",
        (face_image.width * 2 + padding * 3, face_image.height + padding * 2),
        (245, 245, 245),
    )
    canvas.paste(face_image, (padding, padding))
    canvas.paste(face_image, (face_image.width + padding * 2, padding))
    canvas.save(target_path, format="JPEG", quality=92)
    return target_path


def _synthetic_candidate(index: int, cosine_value: float) -> dict[str, Any]:
    embedding = _unit_vector(cosine_value)
    return {
        "embedding_id": index,
        "child_id": index,
        "image_path": f"storage/child_images/synthetic_{index}.jpg",
        "image_hash": f"synthetic_hash_{index}",
        "embedding": json.dumps(embedding, ensure_ascii=True),
        "model_name": "OpenCV-SFace",
        "detector_backend": "OpenCV-YuNet",
        "embedding_dimension": OPENCV_SFACE_EMBEDDING_DIMENSION,
        "quality_score": 90.0,
        "face_confidence": 0.99,
        "case_id": f"SYNTH-{index}",
        "full_name": f"Synthetic Child {index}",
        "age": 10,
        "gender": "Other",
        "guardian_name": "Synthetic Guardian",
        "guardian_phone": "+911234567890",
    }


def _unit_vector(cosine_value: float) -> list[float]:
    clamped = max(-1.0, min(1.0, cosine_value))
    vector = [0.0 for _ in range(OPENCV_SFACE_EMBEDDING_DIMENSION)]
    vector[0] = clamped
    vector[1] = math.sqrt(max(0.0, 1.0 - clamped * clamped))
    return vector


def _child_data(index: int) -> dict[str, object]:
    return {
        "full_name": f"Phase 5 Verification Child {index}",
        "age": 10,
        "gender": "Other",
        "identification_marks": "Phase 5 matching verification sample",
        "last_seen_location": "Phase 5 Verification",
        "last_seen_date": date.today(),
        "last_seen_time": None,
        "description": "Temporary sample registration used to verify Phase 5 matching.",
    }


def _parent_data(index: int) -> dict[str, object]:
    return {
        "guardian_name": f"Phase 5 Guardian {index}",
        "relationship": "Guardian",
        "phone": f"+91987654{index:04d}",
        "email": f"phase5.guardian.{index}@example.com",
        "address": "Phase 5 Verification Address",
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
