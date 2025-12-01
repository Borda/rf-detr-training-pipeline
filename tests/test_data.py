"""Tests for the data module."""

import json
import tempfile
from pathlib import Path

import numpy as np
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


def test_convert_yolo_to_coco():
    """Test converting YOLO dataset to COCO format using supervision package."""
    with tempfile.TemporaryDirectory() as tmpdir:
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
            split_ratios=(0.7, 0.2, 0.1),
            class_names=class_names,
            random_state=42,
        )

        # Verify output directory was created
        assert Path(result).exists()
        assert Path(result) == output_dir

        # Verify splits were created
        train_dir = output_dir / "train"
        valid_dir = output_dir / "valid"
        test_dir = output_dir / "test"

        assert train_dir.exists()
        assert valid_dir.exists()
        assert test_dir.exists()

        # Verify COCO annotations file exists in each split
        train_annotations = train_dir / "_annotations.coco.json"
        valid_annotations = valid_dir / "_annotations.coco.json"
        test_annotations = test_dir / "_annotations.coco.json"

        assert train_annotations.exists()
        assert valid_annotations.exists()
        assert test_annotations.exists()

        # Verify COCO annotation format
        with open(train_annotations) as f:
            train_coco = json.load(f)

        assert "images" in train_coco
        assert "annotations" in train_coco
        assert "categories" in train_coco

        # Verify categories contain the class name
        assert len(train_coco["categories"]) > 0
        assert train_coco["categories"][0]["name"] == "object"

        # Count total images across all splits
        total_images = 0
        for split_dir in [train_dir, valid_dir, test_dir]:
            annotations_path = split_dir / "_annotations.coco.json"
            with open(annotations_path) as f:
                coco_data = json.load(f)
            total_images += len(coco_data["images"])

        # All images should be distributed across splits
        assert total_images == num_images


def test_convert_yolo_to_coco_with_data_yaml():
    """Test converting YOLO dataset with existing data.yaml file."""
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
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
