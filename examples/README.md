# RF-DETR Training Pipeline Examples

This directory contains examples and tutorials for using the RF-DETR training pipeline.

## Installation

To run the examples, install the package with CLI extras:

```bash
pip install -e .[cli,data]
```

This includes dependencies for the command-line interface (jsonargparse) and data downloading (kaggle).

## Dataset Download

Use the local CLI to download datasets. For example, to download the CCTV Weapon Dataset from Kaggle:

```bash
python -m rf_detr_finetuning download kaggle-dataset --name dataset/name --dest data
```

If authentication fails, you'll be prompted for your Kaggle username and API key.

## Dataset Conversion

To convert a YOLO dataset to COCO format with train/valid/test splits:

```bash
python -m rf_detr_finetuning convert yolo-to-coco --input_dir path/to/yolo/dataset --output_dir path/to/output --split_ratios 0.8 0.1 0.1
```

## Real Use Case: Weapon Detection in CCTV Footage

1. Download the dataset as above.
2. Prepare your data by converting to COCO format using the CLI:
   ```bash
   python -m rf_detr_finetuning download kaggle-dataset --name simuletic/cctv-weapon-dataset --dest data
   python -m rf_detr_finetuning convert yolo-to-coco \
      --input_dir data/simuletic/cctv-weapon-dataset/Dataset \
      --output_dir data/cctv-weapon-dataset_coco \
      --class_names '{"0": "person", "1": "weapon"}'
   ```
3. Run training with a config tailored for weapon detection.
   ```bash
   python -m rf_detr_finetuning train --dataset data/cctv-weapon-dataset_coco --config_file config/weapon_detection.yaml
   ```
4. Evaluate the model on test footage.
   ```bash
   python -m rf_detr_finetuning predict --model_path output/checkpoint_best_total.pth --image_path data/simuletic/cctv-weapon-dataset/Dataset/images/Scene1_2.png
   ```

See the main README for module details and configuration options.
