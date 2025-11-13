"""Command-line interface for the RF-DETR training pipeline."""

import logging
import shutil
from pathlib import Path


def download_kaggle_dataset(name: str, dest: str = "data", force: bool = False) -> str:
    """Download a Kaggle dataset into dest using the Kaggle API.

    Requires a valid ~/.kaggle/kaggle.json or environment variables for authentication.
    """
    # Local import to keep dependency usage explicit and avoid import-time failures
    import kagglehub

    kagglehub.login()

    logging.info(f"Starting download of '{name}' into '{dest}'")
    download_path = kagglehub.dataset_download(name, force_download=force)
    logging.info(f"Download complete: {download_path}")

    dest_path = Path(dest)
    dataset_path = dest_path / name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(download_path, dataset_path)
    logging.info(f"Dataaset path: {dataset_path}")
    return str(dataset_path)


commands = {
    "download": {
        "kaggle-dataset": download_kaggle_dataset,
    }
}
