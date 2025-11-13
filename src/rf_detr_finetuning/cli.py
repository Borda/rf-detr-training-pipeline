"""Command-line interface for the RF-DETR training pipeline."""

import logging
import shutil
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import supervision as sv
import yaml

from rf_detr_finetuning.data import convert_yolo_to_coco
from rf_detr_finetuning.finetune import MAP_MODEL_SIZE, finetune_model
from rf_detr_finetuning.predict import prediction


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


def predict(
    image_path: str,
    model_size: Literal[tuple(MAP_MODEL_SIZE.keys())] = "small",
    model_path: str | None = None,
    confidence: float = 0.5,
    class_names: dict[int, str] = None,
) -> None:
    """Predict on an image using a pretrained or checkpoint RF-DETR model and display the result.

    Args:
        image_path: Path to the input image.
        model_size: Size of the RF-DETR model to use.
        model_path: Path to the model checkpoint or pretrained model name.
        confidence: Confidence threshold for predictions.
        class_names: Optional mapping from class id to class name.
    """
    visual = prediction(
        image_path=image_path,
        model_size=model_size,
        model_path=model_path,
        confidence=confidence,
        class_names=class_names,
    )

    # Display the annotated image
    sv.plot_image(visual)


commands = {
    "download": {
        "kaggle-dataset": download_kaggle_dataset,
    },
    "convert": {
        "yolo-to-coco": convert_yolo_to_coco,
    },
    "train": train,
    "predict": predict,
}
