"""Command-line interface for the RF-DETR training pipeline."""

import logging
import random
import shutil
from pathlib import Path
from typing import Literal

from rf_detr_finetuning.data import create_coco_split
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


def convert_yolo_to_coco(
    input_dir: str,
    output_dir: str,
    image_ext: list[str] = [".jpg", ".png"],
    split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    class_names: dict[int, str] = None,
) -> str:
    """Convert a YOLO dataset to COCO format.

    Scans input_dir for images in 'images' subdir and labels in 'labels' subdir, pairs them by filename,
    splits into train/valid/test, and creates COCO annotations in each split dir.

    Args:
        input_dir: Input directory containing 'images' and 'labels' subdirectories.
        output_dir: Output directory for the prepared COCO dataset.
        image_ext: List of image file extensions to include.
        split_ratios: Tuple of (train, valid, test) ratios that sum to 1.0.
        class_names: Optional mapping from class id to class name for COCO categories.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Scanning input directory '{input_path}' and preparing dataset in '{output_path}'")

    image_paths = []
    for ext in image_ext:
        list_imgs = (input_path / "images").glob(f"*{ext}")
        image_paths.extend(list(list_imgs))

    label_paths = [input_path / "labels" / f"{p.stem}.txt" for p in image_paths]
    assert len(label_paths) == len(image_paths), "Mismatch between number of images and labels"

    image_annotation_pairs = list(zip(image_paths, label_paths))
    random.shuffle(image_annotation_pairs)

    assert len(split_ratios) == 3, "Split ratios must be a tuple of three values (train, valid, test)"
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"

    total_files = len(image_annotation_pairs)
    train_count = int(total_files * split_ratios[0])
    valid_count = int(total_files * split_ratios[1])
    test_count = total_files - train_count - valid_count

    splits = [
        ("train", 0, train_count),
        ("valid", train_count, valid_count),
        ("test", train_count + valid_count, test_count),
    ]

    # Create subdirectories for train, valid, and test sets and COCO datasets for each split
    for split_name, split_start, split_len in splits:
        files_split = image_annotation_pairs[split_start : split_start + split_len]
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)
        logging.info(f"{split_name.capitalize()} set: {len(files_split)} images")
        create_coco_split(files_split, str(split_dir), dataset_id_offset=split_start, class_names=class_names)

    logging.info(f"Dataset preparation complete. The prepared dataset is in: {output_path}")
    return str(output_path)


def train(config_file: str, dataset: str, model_size: Literal[tuple(MAP_MODEL_SIZE.keys())] = "small") -> None:
    """Train the RF-DETR model using the provided YAML config and dataset path.

    Args:
        config_file: Path to the YAML training configuration file.
        dataset: Path to the prepared dataset directory.
        model_size: Size of the RF-DETR model to use.
    """
    import yaml

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    finetune_model(model_size=model_size, dataset_path=dataset, config=cfg)


commands = {
    "download": {
        "kaggle-dataset": download_kaggle_dataset,
    },
    "convert": {
        "yolo-to-coco": convert_yolo_to_coco,
    },
    "train": train,
}
