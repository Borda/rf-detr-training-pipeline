"""Command-line interface for the RF-DETR training pipeline."""

import logging
import shutil
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import yaml

from rf_detr_finetuning.data import convert_yolo_to_coco
from rf_detr_finetuning.finetune import MAP_MODEL_SIZE, finetune_model


def download_kaggle_dataset(name: str, dest: str = "data", force: bool = False) -> str:
    """Download a Kaggle dataset into dest using the Kaggle API.

    Args:
        name: Name of the Kaggle dataset to download.
        dest: Destination directory for the downloaded dataset.
        force: Whether to force re-download if the dataset already exists.
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


def train(config_file: str, dataset: str, model_size: Literal[tuple(MAP_MODEL_SIZE.keys())] = "small") -> None:
    """Train the RF-DETR model using the provided YAML config and dataset path.

    Args:
        config_file: Path to the YAML training configuration file.
        dataset: Path to the prepared dataset directory.
        model_size: Size of the RF-DETR model to use.
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    finetune_model(model_size=model_size, dataset_path=dataset, config=cfg)

    # After training, try to display the metrics plot if it exists and GUI is available
    metrics_plot = Path("output/metrics_plot.png")
    if not metrics_plot.exists():
        return
    img = plt.imread(str(metrics_plot))
    plt.imshow(img)
    plt.title("Training Metrics")
    try:
        plt.show()
    except Exception:
        logging.warning("GUI not available, skipping plot display.")


commands = {
    "download": {
        "kaggle-dataset": download_kaggle_dataset,
    },
    "convert": {
        "yolo-to-coco": convert_yolo_to_coco,
    },
    "train": train,
}
