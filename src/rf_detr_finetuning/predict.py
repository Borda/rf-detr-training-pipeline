"""Prediction script for RF-DETR finetuned models."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from supervision import Color

from rf_detr_finetuning.finetune import MAP_MODEL_SIZE


def prediction(
    image_path: str,
    model_size: str,
    model_path: str | None = None,
    confidence: float = 0.5,
    class_names: dict[int, str] = None,
) -> np.ndarray:
    """Predict on an image using a pretrained or checkpoint RF-DETR model.

    Args:
        model_size: Size of the RF-DETR model to use (one of 'base', 'small', 'nano', 'large', 'medium').
        model_path: Path to the model checkpoint or pretrained model name.
        image_path: Path to the input image.
        confidence: Confidence threshold for predictions.
        class_names: Optional mapping from class id to class name.
    """
    assert model_size.lower() in MAP_MODEL_SIZE.keys(), f"Model size must be one of {list(MAP_MODEL_SIZE.keys())}"
    logging.info(f"Loading model from {model_size}")
    ModelClass = MAP_MODEL_SIZE[model_size.lower()]
    # Load the model
    if Path(model_path).exists():
        # Assume it's a checkpoint path
        model = ModelClass(pretrain_weights=model_path)
    else:
        model = ModelClass()

    # Perform inference
    logging.info(f"Processing image: {image_path}")
    predictions = model.predict(image_path, confidence=confidence)
    logging.info(f"{predictions=}")

    # Get labels from predictions
    if not class_names:
        class_names = model.class_names
    labels = [class_names[cls_id] for cls_id in predictions.class_id]

    # Load the image
    image = plt.imread(image_path)[..., :3]
    annotated_image = image[:, :, ::-1]
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, predictions)
    annotated_image = sv.LabelAnnotator(text_color=Color.RED).annotate(annotated_image, predictions, labels=labels)

    return annotated_image
