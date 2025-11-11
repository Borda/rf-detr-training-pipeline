# Tests

This directory contains tests for the RF-DETR fine-tuning pipeline.

## Structure

Tests are organized to mirror the source code structure:

```
tests/
├── test_data.py          # Tests for data module
├── test_training.py      # Tests for training module
├── test_evaluation.py    # Tests for evaluation module
└── test_utils.py         # Tests for utils module
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/rf_detr_finetuning --cov-report=html

# Run specific test file
pytest tests/test_data.py
```
