# RF-DETR Training Pipeline

A modular fine-tuning pipeline for RF-DETR (Real-time DEtection TRansformer) models.

## Project Structure

```
rf-detr-training-pipeline/
├── config/              # Configuration files for training and evaluation
├── src/                 # Source code
│   ├── data/           # Data loading and preprocessing
│   ├── training/       # Training logic and loops
│   ├── evaluation/     # Evaluation metrics and logic
│   └── utils/          # Utility functions
├── scripts/            # Executable scripts
├── examples/           # Example usage and tutorials
└── README.md           # This file
```

## Requirements

- Python 3.10 or higher

## Modules

### src/data

Contains data loading, preprocessing, and augmentation logic for RF-DETR training.

### src/training

Implements training loops, optimizer configuration, and learning rate scheduling.

### src/evaluation

Provides evaluation metrics and validation logic for model assessment.

### src/utils

Contains utility functions for logging, configuration management, and visualization.

## Getting Started

See the `examples/` directory for usage examples and tutorials.

## Configuration

Configuration files are stored in the `config/` directory. See `config/README.md` for details.

## Scripts

Executable scripts for training and evaluation are in the `scripts/` directory. See `scripts/README.md` for details.
