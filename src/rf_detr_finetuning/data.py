"""Utilities for converting annotation files and images into COCO-style datasets.

This module provides helpers to parse label files (supporting multiple common annotation formats and simple
autodetection heuristics) and to assemble a COCO-format dataset (images + annotations + categories). Function docstrings
describe behaviors and expectations without duplicating type information already present in function signatures.

The module leverages the `supervision` package for dataset loading, splitting, and format conversion.

"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import supervision as sv
import yaml


def parse_yolo_annotations(
    label_path: str | Path, image_width: int, image_height: int, annotation_id_counter: int
) -> tuple[list[dict[str, Any]], int]:
    """Convert a YOLO-format label file into COCO-style annotation dicts.

    Each non-empty, non-comment line in the label file is expected to be:
        "<class> <cx> <cy> <w> <h>"
    where coordinates are normalized to [0,1]. Lines that are blank, start with '#',
    or don't have five tokens are skipped with a warning. Coordinates are converted
    to absolute COCO bbox format [x_top_left, y_top_left, width, height]. Annotation
    ids are assigned starting from annotation_id_counter and the counter is advanced.

    Args:
        label_path: path to the YOLO label file.
        image_width: pixel width of the corresponding image.
        image_height: pixel height of the corresponding image.
        annotation_id_counter: starting value for assigning annotation ids.

    Returns:
        A tuple (annotations, new_counter) where annotations is a list of COCO-style
        dicts (image_id left as None) and new_counter is the updated annotation id counter.

    """
    coco_annotations: list[dict[str, Any]] = []
    label_path = Path(label_path)
    try:
        with label_path.open("r") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        logging.warning(f"Annotation file not found for {label_path}. Skipping.")
        return [], annotation_id_counter

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # skip empty/comment lines

        parts = line.split()
        if len(parts) != 5:
            logging.warning(f"Skipping invalid line in {label_path}: {line}")
            continue

        try:
            class_id = int(parts[0])
            center_x_norm, center_y_norm, bbox_width_norm, bbox_height_norm = map(float, parts[1:])
        except ValueError:
            logging.warning(f"Could not parse numbers in {label_path}: {line}")
            continue

        # Convert normalized YOLO to absolute COCO format (x_top_left, y_top_left, width, height)
        x_center = center_x_norm * image_width
        y_center = center_y_norm * image_height
        width_abs = bbox_width_norm * image_width
        height_abs = bbox_height_norm * image_height

        x_top_left = x_center - (width_abs / 2)
        y_top_left = y_center - (height_abs / 2)

        # Clamp bbox to image bounds and ensure non-negative dimensions to avoid invalid COCO annotations
        x_top_left = max(0.0, x_top_left)
        y_top_left = max(0.0, y_top_left)
        width_abs = max(0.0, min(image_width - x_top_left, width_abs))
        height_abs = max(0.0, min(image_height - y_top_left, height_abs))

        # Only add annotation if bbox dimensions are positive
        if width_abs > 0.0 and height_abs > 0.0:
            annotation_id_counter += 1
            coco_annotations.append(
                {
                    "id": annotation_id_counter,
                    "image_id": None,  # set to None here; filled later when image_id is known
                    "category_id": class_id,
                    "bbox": [x_top_left, y_top_left, width_abs, height_abs],
                    "area": width_abs * height_abs,
                    "iscrowd": 0,
                }
            )
        else:
            logging.warning(f"Skipping invalid bounding box with non-positive dimensions in {label_path}: {line}")

    return coco_annotations, annotation_id_counter


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


def convert_yolo_to_coco(
    input_dir: str,
    output_dir: str,
    image_ext: list[str] | None = None,
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
        image_ext: List of image file extensions to include (currently handled by supervision's auto-detection).
        split_ratios: Tuple of (train, valid, test) ratios that sum to 1.0.
        class_names: Optional mapping from class id to class name for COCO categories.
        random_state: Optional seed for reproducible dataset splitting.

    Returns:
        Path to the output directory.

    """
    if image_ext is None:
        image_ext = [".jpg", ".png"]

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Scanning input directory '{input_path}' and preparing dataset in '{output_path}'")

    if len(split_ratios) != 3:
        raise ValueError("Split ratios must be a tuple of three values (train, valid, test)")
    if abs(sum(split_ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    # Check for existing data.yaml or create a temporary one from class_names
    data_yaml_path = input_path / "data.yaml"
    temp_yaml_context = None

    if not data_yaml_path.exists():
        if class_names is None:
            raise ValueError(
                "Either 'data.yaml' must exist in input_dir or 'class_names' must be provided "
                "to specify the class mapping."
            )
        # Create a temporary data.yaml file from class_names
        temp_yaml_context = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        # Convert dict[int, str] to list format expected by supervision
        max_class_id = max(class_names.keys())
        names_list = [class_names.get(i, f"class_{i}") for i in range(max_class_id + 1)]
        yaml_data = {"nc": len(names_list), "names": names_list}
        yaml.dump(yaml_data, temp_yaml_context)
        temp_yaml_context.flush()
        data_yaml_path = Path(temp_yaml_context.name)

    try:
        # Load dataset using supervision
        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=str(input_path / "images"),
            annotations_directory_path=str(input_path / "labels"),
            data_yaml_path=str(data_yaml_path),
        )

        logging.info(f"Loaded {len(dataset)} images with classes: {dataset.classes}")

        # Split dataset into train and remaining (valid + test)
        train_ratio = split_ratios[0]
        remaining_ratio = split_ratios[1] + split_ratios[2]

        if train_ratio > 0 and remaining_ratio > 0:
            train_ds, remaining_ds = dataset.split(
                split_ratio=train_ratio,
                random_state=random_state,
                shuffle=True,
            )
        elif train_ratio > 0:
            train_ds = dataset
            remaining_ds = None
        else:
            train_ds = None
            remaining_ds = dataset

        # Split remaining into valid and test
        if remaining_ds is not None and len(remaining_ds) > 0 and split_ratios[1] > 0 and split_ratios[2] > 0:
            # Calculate the ratio of valid within the remaining portion
            valid_ratio_in_remaining = split_ratios[1] / remaining_ratio
            valid_ds, test_ds = remaining_ds.split(
                split_ratio=valid_ratio_in_remaining,
                random_state=random_state,
                shuffle=True,
            )
        elif remaining_ds is not None and split_ratios[1] > 0:
            valid_ds = remaining_ds
            test_ds = None
        elif remaining_ds is not None and split_ratios[2] > 0:
            valid_ds = None
            test_ds = remaining_ds
        else:
            valid_ds = None
            test_ds = None

        # Export each split to COCO format
        if train_ds is not None and len(train_ds) > 0:
            _export_dataset_split(train_ds, output_path, "train")

        if valid_ds is not None and len(valid_ds) > 0:
            _export_dataset_split(valid_ds, output_path, "valid")

        if test_ds is not None and len(test_ds) > 0:
            _export_dataset_split(test_ds, output_path, "test")

        logging.info(f"Dataset preparation complete. The prepared dataset is in: {output_path}")
        return str(output_path)

    finally:
        # Clean up temporary yaml file if created
        if temp_yaml_context is not None:
            temp_yaml_context.close()
            Path(temp_yaml_context.name).unlink(missing_ok=True)
