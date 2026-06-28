import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATABASE_PATH  # noqa: E402
from services.registration_service import register_missing_child  # noqa: E402


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
    parser = argparse.ArgumentParser(
        description="Register one local sample image and verify a face embedding row is stored."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a clear single-face JPG or PNG child/sample image.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        print(f"FAILED: image file does not exist: {image_path}")
        return 1

    uploaded_file = LocalUploadedFile(path=image_path, type=_content_type_for_path(image_path))

    child_data = {
        "full_name": "Phase 4A Sample Child",
        "age": 10,
        "gender": "Other",
        "identification_marks": "Phase 4A embedding verification sample",
        "last_seen_location": "Phase 4A Verification",
        "last_seen_date": date.today(),
        "last_seen_time": None,
        "description": "Temporary sample registration used to verify embedding storage.",
    }
    parent_data = {
        "guardian_name": "Phase 4A Verifier",
        "relationship": "Guardian",
        "phone": "+911234567890",
        "email": "phase4a.verify@example.com",
        "address": "Phase 4A Verification Address",
        "government_id_type": "",
        "government_id_last4": "",
    }

    try:
        result = register_missing_child(child_data, parent_data, [uploaded_file])
        embedding_count = _count_embeddings(result["child_id"])
    except Exception as exc:
        print(f"FAILED: sample registration did not complete: {exc}")
        return 1

    if embedding_count < 1:
        print(f"FAILED: registration succeeded but no embedding row was stored for {result['case_id']}")
        return 1

    print("OK: sample registration completed and embedding was stored.")
    print(f"Case ID: {result['case_id']}")
    print(f"Child ID: {result['child_id']}")
    print(f"Embedding rows: {embedding_count}")
    return 0


def _count_embeddings(child_id: int) -> int:
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE child_id = ?",
            (child_id,),
        ).fetchone()
    finally:
        connection.close()

    return int(row[0] if row else 0)


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    return "image/jpeg"


if __name__ == "__main__":
    raise SystemExit(main())
