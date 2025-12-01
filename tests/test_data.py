"""Tests for the data module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from rf_detr_finetuning.data import convert_yolo_to_coco, parse_yolo_annotations


def test_parse_yolo_annotations():
    """Test parsing of YOLO annotations."""
    # Create a temporary label file
    label_content = """0 0.5 0.5 0.2 0.2
1 0.1 0.1 0.1 0.1
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(label_content)
        label_path = Path(f.name)

    try:
        annotations, counter = parse_yolo_annotations(label_path, 640, 480, 0)
        assert len(annotations) == 2
        assert annotations[0]["category_id"] == 0
        assert annotations[1]["category_id"] == 1
        assert counter == 2
    finally:
        label_path.unlink()


@pytest.mark.parametrize(
    ("split_ratios", "expected_splits"),
    [
        ((1.0, 0.0, 0.0), ["train"]),
        ((0.0, 1.0, 0.0), ["valid"]),
        ((0.0, 0.0, 1.0), ["test"]),
        ((0.7, 0.2, 0.1), ["train", "valid", "test"]),
        ((0.5, 0.5, 0.0), ["train", "valid"]),
        ((0.0, 0.5, 0.5), ["valid", "test"]),
    ],
)
def test_convert_yolo_to_coco(split_ratios, expected_splits, tmpdir):
    """Test converting YOLO dataset to COCO format with different split configurations."""
    input_dir = Path(tmpdir) / "input"
    output_dir = Path(tmpdir) / "output"

    # Create input directory structure
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create test images and labels
    num_images = 10
    for i in range(num_images):
        # Create a simple test image
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(images_dir / f"img_{i}.jpg")

        # Create a label file with a single annotation
        label_path = labels_dir / f"img_{i}.txt"
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

    # Define class names
    class_names = {0: "object"}

    # Run conversion with fixed random state for reproducibility
    result = convert_yolo_to_coco(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        split_ratios=split_ratios,
        class_names=class_names,
        random_state=42,
    )

    # Verify output directory was created
    assert Path(result).exists()
    assert Path(result) == output_dir

    # Verify expected splits exist and others do not
    all_splits = ["train", "valid", "test"]
    total_images = 0
    for split in all_splits:
        split_dir = output_dir / split
        if split in expected_splits:
            assert split_dir.exists(), f"Expected split '{split}' directory to exist"
            annotations_path = split_dir / "_annotations.coco.json"
            assert annotations_path.exists(), f"Expected annotations file for '{split}' to exist"
            with open(annotations_path) as f:
                coco_data = json.load(f)
            total_images += len(coco_data["images"])

            # Verify COCO annotation format (only for the first expected split to avoid repetition)
            if split == expected_splits[0]:
                assert "images" in coco_data
                assert "annotations" in coco_data
                assert "categories" in coco_data
                assert len(coco_data["categories"]) > 0
                assert coco_data["categories"][0]["name"] == "object"
        else:
            assert not split_dir.exists(), f"Unexpected split '{split}' directory should not exist"

    # All images should be distributed across expected splits
    assert total_images == num_images


def test_convert_yolo_to_coco_with_data_yaml(tmpdir):
    """Test converting YOLO dataset with existing data.yaml file."""
    input_dir = Path(tmpdir) / "input"
    output_dir = Path(tmpdir) / "output"

    # Create input directory structure
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create data.yaml file
    data_yaml = {"nc": 2, "names": ["cat", "dog"]}
    with open(input_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    # Create test images and labels
    for i in range(5):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(images_dir / f"img_{i}.jpg")

        label_path = labels_dir / f"img_{i}.txt"
        with open(label_path, "w") as f:
            f.write(f"{i % 2} 0.5 0.5 0.3 0.3\n")

    # Run conversion without class_names (should use data.yaml)
    result = convert_yolo_to_coco(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        split_ratios=(0.8, 0.2, 0.0),
        random_state=42,
    )

    # Verify output
    assert Path(result).exists()

    # Verify categories from data.yaml are used
    train_annotations = output_dir / "train" / "_annotations.coco.json"
    with open(train_annotations) as f:
        coco_data = json.load(f)

    category_names = [cat["name"] for cat in coco_data["categories"]]
    assert "cat" in category_names
    assert "dog" in category_names
