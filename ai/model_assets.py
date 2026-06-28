import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from utils.logger import get_logger


logger = get_logger(__name__)

DOWNLOAD_RETRY_COUNT = 3
DOWNLOAD_TIMEOUT_SECONDS = 300
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


class ModelAssetError(Exception):
    """Raised when a required local model asset is unavailable."""


@dataclass(frozen=True)
class ModelAssetSpec:
    path: Path
    urls: tuple[str, ...]
    sha256: str
    size_bytes: int


def ensure_model_file(spec: ModelAssetSpec) -> Path:
    if _is_valid_model_file(spec.path, spec.sha256, spec.size_bytes):
        return spec.path

    if spec.path.exists():
        logger.warning("Removing invalid model asset before re-download: %s", spec.path)
        spec.path.unlink()

    spec.path.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for url in spec.urls:
        for attempt in range(1, DOWNLOAD_RETRY_COUNT + 1):
            temporary_path = spec.path.with_suffix(f"{spec.path.suffix}.download")
            try:
                if temporary_path.exists():
                    temporary_path.unlink()

                logger.info(
                    "Downloading model asset %s from %s attempt=%s",
                    spec.path.name,
                    url,
                    attempt,
                )
                _download_file(url, temporary_path)
                _validate_model_file(temporary_path, spec.sha256, spec.size_bytes)
                temporary_path.replace(spec.path)
                logger.info("Model asset ready at %s", spec.path)
                return spec.path
            except ModelAssetError as exc:
                message = f"{url} attempt {attempt}: {exc}"
                errors.append(message)
                logger.warning("Model download attempt failed: %s", message)
                _remove_if_exists(temporary_path)
                if attempt < DOWNLOAD_RETRY_COUNT:
                    time.sleep(min(2**attempt, 8))

    raise ModelAssetError(
        f"Could not download a valid official model asset for {spec.path.name}. "
        + " | ".join(errors[-5:])
    )


def _is_valid_model_file(path: Path, expected_sha256: str, expected_size: int) -> bool:
    if not path.exists() or not path.is_file():
        return False

    try:
        _validate_model_file(path, expected_sha256, expected_size)
    except ModelAssetError as exc:
        logger.warning("Invalid model asset %s: %s", path, exc)
        return False

    return True


def _validate_model_file(path: Path, expected_sha256: str, expected_size: int) -> None:
    if _looks_like_git_lfs_pointer(path):
        raise ModelAssetError(f"{path.name} is a Git LFS pointer, not an ONNX model.")

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ModelAssetError(
            f"{path.name} has size {actual_size} bytes; expected {expected_size} bytes."
        )

    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ModelAssetError(
            f"{path.name} has SHA-256 {actual_sha256}; expected {expected_sha256}."
        )


def _download_file(url: str, target_path: Path) -> None:
    request = Request(
        url,
        headers={
            "Accept": "application/octet-stream,*/*",
            "User-Agent": "missing-child-id-system/phase4a",
        },
    )

    try:
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with target_path.open("wb") as output_file:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    output_file.write(chunk)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ModelAssetError(f"Download failed: {exc}") from exc


def _looks_like_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            header = file.read(128)
    except OSError:
        return False

    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        logger.warning("Could not remove temporary model download: %s", path)
