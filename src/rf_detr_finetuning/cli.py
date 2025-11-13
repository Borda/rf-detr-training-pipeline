"""Command-line interface for the RF-DETR training pipeline."""

import logging
import random
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from .data import create_coco_split


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


def convert_yolo_to_coco(
    input_dir: str,
    output_dir: str,
    image_ext: list[str] = [".jpg", ".png"],
    split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    class_names: dict[int, str] = None,
) -> str:
    """Convert a YOLO dataset to COCO format.

    Scans input_dir for images and labels, pairs them, splits into train/valid/test, and creates COCO annotations.
    image_ext can be a list to include multiple formats like .jpg and .png.
    split_ratios is a tuple of (train, valid, test) ratios that sum to 1.0.
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
    for split_name, split_start, split_len in tqdm(splits):
        files_split = image_annotation_pairs[split_start : split_start + split_len]
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)
        logging.info(f"{split_name.capitalize()} set: {len(files_split)} images")
        create_coco_split(files_split, str(split_dir), dataset_id_offset=split_start, class_names=class_names)

    logging.info(f"Dataset preparation complete. The prepared dataset is in: {output_path}")
    return str(output_path)


commands = {
    "download": {
        "kaggle-dataset": download_kaggle_dataset,
    },
    "convert": {
        "yolo-to-coco": convert_yolo_to_coco,
    },
}
