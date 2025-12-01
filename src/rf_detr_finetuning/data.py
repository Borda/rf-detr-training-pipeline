"""Utilities for converting annotation files and images into COCO-style datasets.

This module provides helpers to parse label files (supporting multiple common annotation formats and simple
autodetection heuristics) and to assemble a COCO-format dataset (images + annotations + categories). Function docstrings
describe behaviors and expectations without duplicating type information already present in function signatures.

The module leverages the `supervision` package for dataset loading, splitting, and format conversion.

"""

import logging
import shutil
import tempfile
from pathlib import Path

import supervision as sv
import yaml
from PIL import Image


def _export_dataset_split(dataset: sv.DetectionDataset, output_dir: Path, split_name: str) -> None:
    """Export a dataset split to COCO format in the output directory.

    Args:
        dataset: The supervision DetectionDataset to export.
        output_dir: The base output directory.
        split_name: Name of the split (e.g., 'train', 'valid', 'test').

    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    dataset.as_coco(
        images_directory_path=str(split_dir),
        annotations_path=str(split_dir / "_annotations.coco.json"),
    )

    logging.info(f"{split_name.capitalize()} set: {len(dataset)} images exported to {split_dir}")


def _create_temp_data_yaml(class_names: dict[int, str]) -> tuple[Path, tempfile.NamedTemporaryFile]:
    """Create a temporary data.yaml file from class_names dict for supervision package.

    Args:
        class_names: Mapping from class id to class name.

    Returns:
        Tuple of (Path to temporary data.yaml, NamedTemporaryFile context).

    """
    temp_yaml_context = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    # Convert dict[int, str] to list format expected by supervision
    max_class_id = max(class_names.keys()) if class_names else 0
    names_list = [class_names.get(i, f"class_{i}") for i in range(max_class_id + 1)]
    yaml_data = {"nc": len(names_list), "names": names_list}
    yaml.dump(yaml_data, temp_yaml_context)
    temp_yaml_context.flush()
    data_yaml_path = Path(temp_yaml_context.name)
    return data_yaml_path, temp_yaml_context


def convert_yolo_to_coco(
    input_dir: str,
    output_dir: str,
    image_ext: tuple[str] | list[str] | str = (".jpg", ".jpeg", ".png"),
    split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    class_names: dict[int, str] | None = None,
    random_state: int | None = None,
) -> str:
    """Convert a YOLO dataset to COCO format using the supervision package.

    Scans input_dir for images in 'images' subdir and labels in 'labels' subdir, pairs them by filename,
    splits into train/valid/test, and creates COCO annotations in each split dir.

    Args:
        input_dir: Input directory containing 'images' and 'labels' subdirectories.
        output_dir: Output directory for the prepared COCO dataset.
        image_ext: Deprecated parameter kept for backward compatibility. Image file detection is now
            handled automatically by the supervision package.
        split_ratios: Tuple of (train, valid, test) ratios that sum to 1.0.
        class_names: Optional mapping from class id to class name for COCO categories.
            Required if 'data.yaml' does not exist in input_dir.
        random_state: Optional seed for reproducible dataset splitting.

    Returns:
        Path to the output directory.

    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Scanning input directory '{input_path}' and preparing dataset in '{output_path}'")
    if isinstance(image_ext, str):
        image_ext = [image_ext]

    if split_ratios is not None:
        if len(split_ratios) not in (1, 2, 3):
            raise ValueError("split_ratios must be None or a tuple of 1, 2, or 3 floats")
        if len(split_ratios) == 3 and abs(sum(split_ratios) - 1.0) > 1e-9:
            raise ValueError("split_ratios must sum to 1.0")

    # Check for existing data.yaml or create a temporary one from class_names
    data_yaml_path = input_path / "data.yaml"
    temp_yaml_context = None

    if not data_yaml_path.exists():
        data_yaml_path, temp_yaml_context = _create_temp_data_yaml(class_names)

    # Preprocess images: convert RGBA to RGB
    temp_images_dir = tempfile.mkdtemp()
    temp_images_path = Path(temp_images_dir)
    img_paths = []
    for ext in image_ext:
        img_paths.extend((input_path / "images").glob(f"*.{ext.lstrip('.')}"))
    for img_path in img_paths:
        with Image.open(img_path) as img:
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(temp_images_path / img_path.name)

    # Load dataset using supervision
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=str(temp_images_path),
        annotations_directory_path=str(input_path / "labels"),
        data_yaml_path=str(data_yaml_path),
    )

    logging.info(f"Loaded {len(dataset)} images with classes: {dataset.classes}")

    # Normalize split_ratios
    if split_ratios is None:
        train_ratio, valid_ratio, test_ratio = 1.0, 0.0, 0.0
    elif len(split_ratios) == 1:
        train_ratio = split_ratios[0]
        valid_ratio = 1.0 - train_ratio
        test_ratio = 0.0
    elif len(split_ratios) == 2:
        train_ratio, valid_ratio = split_ratios
        test_ratio = 1.0 - train_ratio - valid_ratio
    else:
        train_ratio, valid_ratio, test_ratio = split_ratios

    # Split dataset into train, valid, and test based on split_ratios
    if train_ratio > 0:
        train_ds, temp_ds = dataset.split(split_ratio=train_ratio, random_state=random_state, shuffle=True)
    else:
        train_ds = None
        temp_ds = dataset

    # Split remaining data into valid and test
    if valid_ratio > 0 and temp_ds is not None:
        valid_ratio_in_temp = valid_ratio / (valid_ratio + test_ratio) if test_ratio > 0 else 1.0
        valid_ds, test_ds = temp_ds.split(split_ratio=valid_ratio_in_temp, random_state=random_state, shuffle=True)
    else:
        valid_ds = temp_ds if valid_ratio > 0 else None
        test_ds = temp_ds if test_ratio > 0 and valid_ratio == 0 else None

    # Export each split to COCO format
    for split_name, split_ds in [("train", train_ds), ("valid", valid_ds), ("test", test_ds)]:
        if split_ds is not None and len(split_ds) > 0:
            _export_dataset_split(split_ds, output_path, split_name)

    # Clean up temporary files
    shutil.rmtree(temp_images_dir)
    if temp_yaml_context is not None:
        temp_yaml_context.close()
        data_yaml_path.unlink()

    logging.info(f"Dataset preparation complete. The prepared dataset is in: {output_path}")
    return str(output_path)
