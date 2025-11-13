# RF-DETR Training Pipeline Examples

This directory contains examples and tutorials for using the RF-DETR training pipeline.

## Installation

To run the examples, install the package with CLI extras:

```bash
pip install -e .[cli,data]
```

This includes dependencies for the command-line interface (tyro) and data downloading (kaggle).

## Dataset Download

Use the local CLI to download datasets. For example, to download the CCTV Weapon Dataset from Kaggle:

```bash
python -m rf_detr_finetuning download kaggle-dataset --name simuletic/cctv-weapon-dataset --dest data
```

If authentication fails, you'll be prompted for your Kaggle username and API key.

## Real Use Case: Weapon Detection in CCTV Footage

1. Download the dataset as above.
1. Prepare your data (e.g., convert to COCO format using the data module).
1. Run training with a config tailored for weapon detection.
1. Evaluate the model on test footage.

See the main README for module details and configuration options.
