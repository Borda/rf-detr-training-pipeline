"""Finetuning utilities for RF-DETR models."""

import logging

import torch
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

MAP_MODEL_SIZE = {
    "base": RFDETRBase,
    "small": RFDETRSmall,
    "nano": RFDETRNano,
    "large": RFDETRLarge,
    "medium": RFDETRMedium,
}


def finetune_model(model_size: str, dataset_path: str, config: dict) -> dict:
    """Finetune an RF-DETR model on a custom dataset.

    Loads the specified model size, updates the config with dataset path and device,
    and starts the training process. Results are logged and printed.

    Args:
        model_size: Size of the RF-DETR model to use (one of 'base', 'small', 'nano', 'large', 'medium').
        dataset_path: Path to the dataset directory containing images and annotations.
        config: Dictionary of training configuration parameters.
    """
    assert model_size.lower() in MAP_MODEL_SIZE.keys(), f"Model size must be one of {list(MAP_MODEL_SIZE.keys())}"
    logging.info(f"Loading model from {model_size}")
    ModelClass = MAP_MODEL_SIZE[model_size.lower()]
    model = ModelClass()

    logging.info(f"Updating config for training with dataset at {dataset_path}")
    config["dataset_dir"] = dataset_path
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # Start training
    results = model.train(**config)

    # The training results will typically be saved in the specified project/name directory.
    # You can inspect the 'results' object or the output directory for metrics and checkpoints.
    logging.debug(f"Results:\n{results}")
    print(results)
