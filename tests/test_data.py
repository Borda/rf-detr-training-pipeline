"""Tests for the data module."""

import tempfile
from pathlib import Path

from rf_detr_finetuning.data import parse_yolo_annotations


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
