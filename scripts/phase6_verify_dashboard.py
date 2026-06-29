import argparse
import csv
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATABASE_PATH, TEMP_DIR  # noqa: E402
from database.schema import initialize_database  # noqa: E402
from services.dashboard_service import (  # noqa: E402
    delete_child_case,
    export_child_records_csv,
    export_search_history_csv,
    get_child_case_details,
    get_dashboard_overview,
    get_recent_registrations,
    get_recent_search_history,
    search_dashboard_child_records,
)
from services.matching_service import search_found_child  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 6 admin dashboard and analytics.")
    parser.add_argument(
        "--image",
        help="Optional clear single-face JPG/PNG image. If omitted, an existing registered child image is reused.",
    )
    parser.add_argument(
        "--skip-regression-scripts",
        action="store_true",
        help="Skip Phase 4A, Phase 4B, and Phase 5 regression script execution.",
    )
    args = parser.parse_args()

    try:
        initialize_database()
        source_image = _resolve_source_image(args.image)
        work_dir = _make_work_dir()

        print(f"Using source image: {source_image}")
        print(f"Verification workspace: {work_dir}")

        _run_py_compile()

        registered_child = _register_dashboard_sample(source_image, work_dir)
        query_path = _materialize_unique_image(source_image, work_dir / "phase6_query.jpg", marker_index=70)
        search_result = search_found_child(LocalUploadedFile(query_path, "image/jpeg"))

        _verify_dashboard_loads()
        _verify_statistics()
        _verify_recent_registrations(registered_child["case_id"])
        _verify_search_history_retrieval(search_result["search_id"])
        _verify_filters(registered_child)
        _verify_case_details(registered_child)
        _verify_csv_exports(registered_child["case_id"], search_result["search_id"])
        _verify_delete_functionality(registered_child)

        if not args.skip_regression_scripts:
            _run_existing_regression_checks(source_image, work_dir)

    except Exception as exc:
        print(f"FAILED: Phase 6 dashboard verification did not complete: {exc}")
        return 1

    print("OK: Phase 6 dashboard verification passed.")
    return 0


def _run_py_compile() -> None:
    files = [
        "app.py",
        "ai/embedding_generator.py",
        "ai/model_assets.py",
        "config/constants.py",
        "config/settings.py",
        "database/connection.py",
        "database/schema.py",
        "database/repositories/child_repository.py",
        "database/repositories/dashboard_repository.py",
        "database/repositories/embedding_repository.py",
        "database/repositories/image_repository.py",
        "database/repositories/log_repository.py",
        "database/repositories/parent_repository.py",
        "database/repositories/record_repository.py",
        "database/repositories/search_history_repository.py",
        "services/dashboard_service.py",
        "services/embedding_service.py",
        "services/matching_service.py",
        "services/records_service.py",
        "services/registration_service.py",
        "ui/dashboard_ui.py",
        "ui/records_ui.py",
        "ui/registration_ui.py",
        "ui/search_ui.py",
        "utils/file_handler.py",
        "utils/logger.py",
        "utils/validators.py",
        "scripts/phase4a_health_check.py",
        "scripts/phase4a_sample_registration.py",
        "scripts/phase4b_verify_registration.py",
        "scripts/phase5_verify_matching.py",
        "scripts/phase6_verify_dashboard.py",
    ]
    _run_command([sys.executable, "-m", "py_compile", *files])
    print("OK: py_compile passed for Phase 6 and existing project modules.")


def _register_dashboard_sample(source_image: Path, work_dir: Path) -> dict[str, Any]:
    image_path = _materialize_unique_image(source_image, work_dir / "phase6_child.jpg", marker_index=60)
    result = register_missing_child(
        _child_data(),
        _parent_data(),
        [LocalUploadedFile(image_path, _content_type_for_path(image_path))],
    )

    if result.get("embedding_count", 0) < 1:
        raise RuntimeError("Phase 6 sample registration did not store an embedding.")

    case_details = get_child_case_details(result["child_id"])
    print(f"OK: disposable dashboard sample registered case_id={result['case_id']}.")
    return case_details


def _verify_dashboard_loads() -> None:
    overview = get_dashboard_overview()
    required_keys = {"statistics", "recent_registrations", "recent_search_history"}
    if set(overview) != required_keys:
        raise RuntimeError("Dashboard overview did not return the expected sections.")

    print("OK: dashboard overview loads.")


def _verify_statistics() -> None:
    overview = get_dashboard_overview()
    statistics = overview["statistics"]

    connection = sqlite3.connect(DATABASE_PATH)
    try:
        expected_children = _count(connection, "SELECT COUNT(*) FROM missing_children")
        expected_embeddings = _count(connection, "SELECT COUNT(*) FROM face_embeddings")
        expected_searches = _count(connection, "SELECT COUNT(*) FROM search_history")
        expected_successes = _count(
            connection,
            """
            SELECT COUNT(*)
            FROM search_history
            WHERE status = 'completed'
              AND matches_found > 0
            """,
        )
    finally:
        connection.close()

    expected_unsuccessful = max(expected_searches - expected_successes, 0)
    expected_percentage = round((expected_successes / expected_searches) * 100.0, 2) if expected_searches else 0.0

    expected = {
        "total_registered_children": expected_children,
        "total_face_embeddings": expected_embeddings,
        "total_search_requests": expected_searches,
        "total_successful_matches": expected_successes,
        "total_unsuccessful_searches": expected_unsuccessful,
        "match_success_percentage": expected_percentage,
    }
    if statistics != expected:
        raise RuntimeError(f"Dashboard statistics mismatch. expected={expected} actual={statistics}")

    print("OK: dashboard statistics match database counts.")


def _verify_recent_registrations(case_id: str) -> None:
    recent_registrations = get_recent_registrations(limit=20)
    if not recent_registrations:
        raise RuntimeError("Recent registrations retrieval returned no rows after sample registration.")
    if not any(record["case_id"] == case_id for record in recent_registrations):
        raise RuntimeError("Recent registrations did not include the Phase 6 sample case.")

    print("OK: recent registrations retrieval works.")


def _verify_search_history_retrieval(search_id: int) -> None:
    recent_searches = get_recent_search_history(limit=20)
    if not recent_searches:
        raise RuntimeError("Search history retrieval returned no rows after sample search.")
    if not any(record["search_id"] == search_id for record in recent_searches):
        raise RuntimeError("Recent search history did not include the Phase 6 sample search.")

    print("OK: search history retrieval works.")


def _verify_filters(case_details: dict[str, Any]) -> None:
    filter_cases = [
        {"case_id": case_details["case_id"]},
        {"child_name": "Phase 6 Verification Child"},
        {"gender": case_details["gender"]},
        {"age": case_details["age"]},
        {"registration_date": str(case_details["registration_date"])[:10]},
        {
            "case_id": case_details["case_id"],
            "child_name": "Phase 6 Verification Child",
            "gender": case_details["gender"],
            "age": case_details["age"],
            "registration_date": str(case_details["registration_date"])[:10],
        },
    ]

    for filters in filter_cases:
        records = search_dashboard_child_records(filters)
        if not any(record["case_id"] == case_details["case_id"] for record in records):
            raise RuntimeError(f"Dashboard filters did not return the sample case for filters={filters}")

    try:
        search_dashboard_child_records({"age": 99})
    except ValidationError:
        pass
    else:
        raise RuntimeError("Invalid age filter was not rejected.")

    print("OK: dashboard filters work and reject invalid input.")


def _verify_case_details(case_details: dict[str, Any]) -> None:
    loaded = get_child_case_details(case_details["child_id"])
    if loaded["case_id"] != case_details["case_id"]:
        raise RuntimeError("Case details returned the wrong case.")
    if not loaded["images"]:
        raise RuntimeError("Case details did not include uploaded images.")
    if not loaded["embeddings"] or loaded["embedding_status"] != "Stored":
        raise RuntimeError("Case details did not include embedding status.")
    if not loaded["guardian_name"] or not loaded["guardian_phone"]:
        raise RuntimeError("Case details did not include guardian details.")

    print("OK: case details include child, guardian, images, embeddings, and registration date.")


def _verify_csv_exports(case_id: str, search_id: int) -> None:
    child_csv = export_child_records_csv({"case_id": case_id}).decode("utf-8")
    child_rows = list(csv.DictReader(StringIO(child_csv)))
    if not child_rows or child_rows[0]["case_id"] != case_id:
        raise RuntimeError("Child records CSV export did not include the sample case.")

    search_csv = export_search_history_csv().decode("utf-8")
    search_rows = list(csv.DictReader(StringIO(search_csv)))
    if not any(int(row["search_id"]) == search_id for row in search_rows if row.get("search_id")):
        raise RuntimeError("Search history CSV export did not include the sample search.")

    print("OK: child records and search history CSV exports work.")


def _verify_delete_functionality(case_details: dict[str, Any]) -> None:
    child_id = int(case_details["child_id"])
    case_id = str(case_details["case_id"])

    try:
        delete_child_case(child_id, "WRONG-CASE-ID")
    except ValidationError:
        pass
    else:
        raise RuntimeError("Delete confirmation did not reject a wrong Case ID.")

    result = delete_child_case(child_id, case_id)
    if not result.get("deleted"):
        raise RuntimeError("Delete service did not report success.")

    connection = sqlite3.connect(DATABASE_PATH)
    try:
        remaining_child = _count(connection, "SELECT COUNT(*) FROM missing_children WHERE id = ?", (child_id,))
        remaining_images = _count(connection, "SELECT COUNT(*) FROM child_images WHERE child_id = ?", (child_id,))
        remaining_embeddings = _count(connection, "SELECT COUNT(*) FROM face_embeddings WHERE child_id = ?", (child_id,))
    finally:
        connection.close()

    if remaining_child or remaining_images or remaining_embeddings:
        raise RuntimeError("Delete did not remove the child record and dependent rows.")

    print("OK: delete functionality removes the selected case after confirmation.")


def _run_existing_regression_checks(source_image: Path, work_dir: Path) -> None:
    _run_command([sys.executable, "scripts/phase4a_health_check.py"])
    _run_command([sys.executable, "scripts/phase4b_verify_registration.py", "--image", str(source_image)])
    phase5_query = _materialize_unique_image(source_image, work_dir / "phase6_phase5_regression.jpg", marker_index=80)
    phase5_result = search_found_child(LocalUploadedFile(phase5_query, "image/jpeg"))
    if phase5_result["candidate_count"] <= 0 or phase5_result["matches_found"] <= 0:
        raise RuntimeError("Phase 5 matching regression did not return any matches.")
    print("OK: Phase 4A, Phase 4B, and Phase 5 regression checks passed.")


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            f"{' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


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
        raise RuntimeError("No existing child image found. Register one child before running Phase 6 verification.")
    return candidates[0]


def _make_work_dir() -> Path:
    work_dir = TEMP_DIR / "phase6_verification_inputs" / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _materialize_unique_image(source_image: Path, target_path: Path, marker_index: int) -> Path:
    with Image.open(source_image) as source:
        image = source.convert("RGB")

    seed = int(uuid.uuid4().hex[:6], 16)
    draw = ImageDraw.Draw(image)
    marker_color = (
        (seed + marker_index * 23) % 255,
        (seed + marker_index * 41) % 255,
        (seed + marker_index * 67) % 255,
    )
    marker_x = seed % 24
    marker_y = (seed // 24) % 24
    draw.rectangle((marker_x, marker_y, marker_x + 4, marker_y + 4), fill=marker_color)
    image.save(target_path, format="JPEG", quality=92)
    return target_path


def _child_data() -> dict[str, object]:
    return {
        "full_name": "Phase 6 Verification Child",
        "age": 11,
        "gender": "Other",
        "identification_marks": "Phase 6 dashboard verification sample",
        "last_seen_location": "Phase 6 Verification",
        "last_seen_date": date.today(),
        "last_seen_time": None,
        "description": "Temporary sample registration used to verify Phase 6 dashboard behavior.",
    }


def _parent_data() -> dict[str, object]:
    return {
        "guardian_name": "Phase 6 Guardian",
        "relationship": "Guardian",
        "phone": "+919999990006",
        "email": "phase6.guardian@example.com",
        "address": "Phase 6 Verification Address",
        "government_id_type": "",
        "government_id_last4": "",
    }


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    return "image/jpeg"


def _count(connection: sqlite3.Connection, query: str, parameters: tuple[Any, ...] = ()) -> int:
    row = connection.execute(query, parameters).fetchone()
    return int(row[0] if row else 0)


if __name__ == "__main__":
    raise SystemExit(main())
