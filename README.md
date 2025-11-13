# RF-DETR Training Pipeline

A modular fine-tuning pipeline for [RF-DETR](https://github.com/roboflow/rf-detr) (Real-time DEtection TRansformer) models.

## Introduction 📖

The RF-DETR Training Pipeline simplifies the process of fine-tuning [RF-DETR](https://github.com/roboflow/rf-detr) (Real-time DEtection TRansformer) models for object detection tasks. It provides a user-friendly command-line interface (CLI) that handles everything from dataset preparation to model training and inference, enabling users to leverage state-of-the-art detection capabilities without deep expertise in machine learning pipelines.

## Motivation 💡

Fine-tuning large vision models like [RF-DETR](https://github.com/roboflow/rf-detr) can be daunting due to the need for data preprocessing, configuration management, and training orchestration. This pipeline bridges that gap by offering a modular, CLI-driven approach that democratizes access to advanced object detection. Whether you're a researcher prototyping new ideas or a developer integrating detection into applications, the pipeline reduces boilerplate and accelerates iteration.

## Features ✨

This pipeline combines powerful functionality with developer-friendly design to streamline your RF-DETR fine-tuning workflow. Here's what makes it stand out:

- **Simple CLI Interface**: Easy-to-use command-line tools for all pipeline stages
- **Dataset Management**: Download datasets from [Kaggle](https://www.kaggle.com/) with built-in authentication (see the [Kaggle API docs](https://www.kaggle.com/docs/api))
- **Format Conversion**: Automatic conversion from YOLO to COCO format with customizable splits
- **Flexible Training**: YAML-based configuration for reproducible training runs
- **Auto Device Detection**: Automatically selects GPU/CPU based on availability
- **Modular Architecture**: Clean separation of concerns for easy extension and maintenance
- **Type Safety**: Full type hints throughout the codebase
- **Testing**: Unit tests with [pytest](https://docs.pytest.org/en/stable/) and coverage tracking
- **CI/CD**: Automated testing via [GitHub Actions](https://docs.github.com/actions)
- **Pre-commit Hooks**: Automatic code quality checks with [pre-commit](https://pre-commit.com/) and [ruff](https://docs.astral.sh/ruff/)

## Approach 🎯

The pipeline architecture is built around three core principles: simplicity, reliability, and extensibility. Each component is designed to work independently while integrating seamlessly into the complete workflow.

### 1. Data Pipeline

The pipeline implements a robust data handling system:

- **Download Module**: Integrates with the [Kaggle API](https://www.kaggle.com/docs/api) for seamless dataset acquisition
- **Format Conversion**: Parses YOLO annotations and converts to COCO format with proper validation
- **Smart Splitting**: Configurable train/valid/test splits with shuffling
- **Bbox Validation**: Ensures bounding boxes are within image bounds and have positive dimensions

### 2. Configuration Management

- **YAML-based configs**: Human-readable training configurations (see [config/README.md](config/README.md))
- **Automatic overrides**: CLI arguments override config file values
- **Device auto-detection**: Automatically selects CUDA if available, falls back to CPU

### 3. CLI Design

- **Hierarchical commands**: Organized as `python -m rf_detr_finetuning <command> <subcommand>`
  - **Type validation**: Uses [jsonargparse](https://jsonargparse.readthedocs.io/en/stable/) for automatic type checking and help generation
  - **Clear documentation**: Docstrings automatically converted to CLI help messages

## Project Structure 📁

The repository is organized to promote clarity and maintainability. Source code, tests, configuration, and documentation are cleanly separated into dedicated directories.

```
rf-detr-training-pipeline/
├── .github/                       # GitHub templates and workflows
│   ├── workflows/                 # CI/CD workflows
│   ├── ISSUE_TEMPLATE/            # Issue templates
│   ├── PULL_REQUEST_TEMPLATE.md   # PR template
│   └── CONTRIBUTING.md            # Contribution guidelines
├── config/                        # Training configuration files
├── data/                          # Downloaded and prepared datasets
├── src/
│   └── rf_detr_finetuning/        # Main package
├── tests/                         # Test suite
├── examples/                      # Usage tutorials and demos
├── pyproject.toml                 # Project metadata and dependencies
├── .pre-commit-config.yaml        # Pre-commit hooks configuration
└── README.md                      # Top-level documentation
```

## Installation 📦

Getting started is straightforward. Install the package in editable mode with the extras that match your needs.

```bash
pip install -e .[cli,data]
```

<details>
<summary>Development Installation</summary>

```bash
pip install -e .[dev,cli,data]
pre-commit install
```

</details>

## Quick Start 🚀

Follow these steps to get up and running with RF-DETR fine-tuning in minutes. The workflow covers dataset acquisition, preparation, training, and inference.

### Dataset Download

Use the local CLI to download datasets. For example, to download the CCTV Weapon Dataset from Kaggle:

```bash
python -m rf_detr_finetuning download kaggle-dataset --name dataset/name --dest data
```

If authentication fails, you'll be prompted for your Kaggle username and API key.

### Dataset Conversion

To convert a YOLO dataset to COCO format with train/valid/test splits:

```bash
python -m rf_detr_finetuning convert yolo-to-coco \
  --input_dir path/to/yolo/dataset \
  --output_dir path/to/output \
  --split_ratios 0.8 0.1 0.1 \
  --class_names '{"0": "person", "1": "weapon"}'
```

### Training

To train the model on the prepared dataset:

```bash
python -m rf_detr_finetuning train \
  --config config/sample_config.yaml \
  --dataset data/my-dataset_coco
```

The device (GPU/CPU) is automatically detected and set.

![cctv-weapon-train-rf-detr-detection_metrics.png](examples/cctv-weapon-train-rf-detr_metrics.png)

### Prediction

Once training is complete, run inference on new images using your trained model:

```bash
python -m rf_detr_finetuning predict \
  --model_path output/best_checkpoint.pth \
  --image_path path/to/test/image.jpg \
  --confidence 0.5
```

This will display the detected objects with bounding boxes and confidence scores.

## Real Use Case 🎥

To illustrate the pipeline's capabilities, we've included a complete example using the CCTV Weapon Detection dataset. See [examples/README.md](examples/README.md) for a complete walkthrough of weapon detection in CCTV footage using this pipeline.

![cctv-weapon_samples.png](examples/cctv-weapon_samples.png)

## What You Get 🎁

This pipeline provides a complete solution for RF-DETR fine-tuning:

- **End-to-end workflow**: From dataset download to trained model
- **Production-ready code**: Type-safe, tested, and documented
- **Easy customization**: YAML configs and modular design make it simple to adapt
- **Community standards**: Following best practices with CI/CD, pre-commit hooks, and contribution guidelines

The training implementation is ready for extension - currently includes the CLI structure and data pipeline, with the actual RF-DETR training logic ready to be integrated based on your specific use case.

## Configuration ⚙️

All training parameters are managed through YAML configuration files, making it easy to version control your experiments and reproduce results. Configuration files are stored in the `config/` directory. See [config/README.md](config/README.md) for details.

<details>
<summary>Example configuration</summary>

```yaml
epochs: 20
batch_size: 4
imgsz: 640
device: "cuda"  # Auto-detected by CLI
workers: 2
optimizer: "AdamW"
lr: 0.0001
```

</details>

This project is open source and available under the MIT License — see the [LICENSE](./LICENSE) file for details.

## Contributing 🤝

Thank you for considering contributing! This section gives a short, top-level summary; please follow the full guide for detailed steps.

- Summary: file issues for bugs or feature requests, fork the repo, create a feature branch, run tests and style checks, open a pull request for review.
- Quick checklist:
  - Read the full contribution guide: [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)
  - Follow the Code of Conduct: [.github/CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)
  - Run linters and pre-commit hooks locally (pre-commit)
  - Run tests: `pytest -v tests/`
  - Keep changes small and focused; reference related issues in PRs
- CI: All PRs run automated tests and linters via GitHub Actions — ensure CI is green before requesting review.

For the authoritative, detailed process (branching, commit style, review flow, release notes), see the full guide at .github/CONTRIBUTING.md.

## License 📄

This project is open source and available under the MIT License - see LICENSE file for details.

## Acknowledgements 🙏

- [RF-DETR](https://github.com/roboflow/rf-detr) team for the excellent detection model
- [Roboflow RF-DETR training guide](https://rfdetr.roboflow.com/learn/train/) and [Roboflow](https://roboflow.com/) for documentation and training resources
- Contributors and the open-source community
