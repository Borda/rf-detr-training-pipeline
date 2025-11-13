"""RF-DETR Fine-tuning Pipeline.

This package provides a modular pipeline for fine-tuning RF-DETR models.
"""

__version__ = "0.1.0"


from rf_detr_finetuning.data import convert_yolo_to_coco
from rf_detr_finetuning.finetune import finetune_model
from rf_detr_finetuning.predict import prediction

__all__ = [
    "convert_yolo_to_coco",
    "finetune_model",
    "prediction",
]
