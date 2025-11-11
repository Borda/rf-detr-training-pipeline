# RF-DETR Training Pipeline

A modular fine-tuning pipeline for RF-DETR (Real-time DEtection TRansformer) models.

## Project Structure

```
rf-detr-training-pipeline/
├── config/                        # Configuration files for training and evaluation
├── src/
│   └── rf_detr_finetuning/       # Main package
│       ├── data/                  # Dataset loading and preprocessing
│       ├── training/              # Training loops and optimization
│       ├── evaluation/            # Metrics and validation
│       └── utils/                 # Logging, config, visualization
├── scripts/                       # Executable training/inference scripts
├── examples/                      # Usage tutorials and demos
├── tests/                         # Test suite
└── README.md                      # This file
```

## Requirements

- Python 3.10 or higher

## Modules

### rf_detr_finetuning.data

Contains data loading, preprocessing, and augmentation logic for RF-DETR training.

### rf_detr_finetuning.training

Implements training loops, optimizer configuration, and learning rate scheduling.

### rf_detr_finetuning.evaluation

Provides evaluation metrics and validation logic for model assessment.

### rf_detr_finetuning.utils

Contains utility functions for logging, configuration management, and visualization.

## Getting Started

See the `examples/` directory for usage examples and tutorials.

## Configuration

Configuration files are stored in the `config/` directory. See `config/README.md` for details.

## Scripts

Executable scripts for training and evaluation are in the `scripts/` directory. See `scripts/README.md` for details.

## Testing

Tests are located in the `tests/` directory. See `tests/README.md` for details on running tests.
