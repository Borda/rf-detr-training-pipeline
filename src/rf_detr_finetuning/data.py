"""Utilities for converting annotation files and images into COCO-style datasets.

This module provides helpers to parse label files (supporting multiple common
annotation formats and simple autodetection heuristics) and to assemble a
COCO-format dataset (images + annotations + categories). Function docstrings
describe behaviors and expectations without duplicating type information already
present in function signatures.
"""

import datetime
import json
import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2


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


def create_coco_split(
    file_list: Iterable[tuple[str | Path, str | Path]],
    output_dir: str | Path,
    dataset_id_offset: int,
    class_names: dict[int, str],
) -> None:
    """Create a COCO-style dataset split from image/YOLO-label pairs.

    For each (image_path, label_path) pair the image is read to obtain dimensions,
    YOLO annotations are parsed and converted to COCO format, images are copied into
    output_dir, and a single JSON file "_annotations.coco.json" is written with the
    collected images, annotations and categories. The output directory is created
    if it does not exist. Images that cannot be read are skipped with a warning.

    Args:
        file_list: iterable of (image_path, label_path) tuples.
        output_dir: destination directory for images and the annotations JSON.
        dataset_id_offset: offset added to generated image ids (useful to avoid id collisions).
        class_names: mapping from class id to class name used to populate categories.

    Returns:
        None. Produces files on disk (copied images and the annotations JSON).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = class_names or {}

    coco_data: dict[str, Any] = {
        "info": {
            "description": "Dataset in COCO format",
            "version": "0.1.0",
            "year": datetime.datetime.now().year,
            "contributor": "Generated by ...",
            "date_created": datetime.datetime.now(datetime.timezone.utc).isoformat(" "),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Add categories based on provided class names
    for class_id, class_name in class_names.items():
        coco_data["categories"].append({"id": class_id, "name": class_name, "supercategory": ""})

    annotation_id_counter = 0

    for image_path, label_path in file_list:
        image_path = Path(image_path)
        # Read image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning(f"Could not read image {image_path} for COCO conversion.")
            continue

        height, width = image.shape[:2]
        image_id = len(coco_data["images"]) + dataset_id_offset

        coco_data["images"].append({"id": image_id, "width": width, "height": height, "file_name": image_path.name})

        # Parse YOLO annotations and convert to COCO format
        image_coco_annotations, annotation_id_counter = parse_yolo_annotations(
            label_path, width, height, annotation_id_counter
        )

        for anno in image_coco_annotations:
            anno["image_id"] = image_id
            coco_data["annotations"].append(anno)

        # Copy image file to the output directory
        shutil.copy(str(image_path), str(output_dir / image_path.name))

    # Save the COCO annotations to a JSON file
    with open(output_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco_data, f, indent=4)

    logging.info(
        f"Created COCO dataset in {output_dir}"
        f" with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations."
    )
