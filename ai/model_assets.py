from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

from utils.logger import get_logger


logger = get_logger(__name__)


class ModelAssetError(Exception):
    """Raised when a required local model asset is unavailable."""


def ensure_model_file(model_path: Path, model_url: str) -> Path:
    if model_path.exists() and model_path.is_file() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".download")

    try:
        logger.info("Downloading model asset from %s", model_url)
        urlretrieve(model_url, temporary_path)
        if temporary_path.stat().st_size <= 0:
            raise ModelAssetError(f"Downloaded model asset is empty: {model_path.name}")
        temporary_path.replace(model_path)
        logger.info("Model asset ready at %s", model_path)
        return model_path
    except (OSError, URLError) as exc:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                logger.warning("Could not remove incomplete model download: %s", temporary_path)
        raise ModelAssetError(
            f"Could not prepare model asset {model_path.name}. Check internet access and storage permissions."
        ) from exc
