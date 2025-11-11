# Agent HQ Configuration for RF-DETR Fine-Tuning (Roboflow Stack)

## 🧠 Agents

### - engineer
**Role**: RF-DETR fine-tuning and deployment specialist  
**Tools**: Python, Roboflow SDK, `rfdetr`, `supervision`, Weights & Biases  
**Behavior**:
- Review PRs modifying `train.py`, `config.yaml`, or `inference.py`
- Validate dataset loading via Roboflow API and class mappings
- Ensure correct use of `optimize_for_inference()` and model export
- Check reproducibility: fixed seeds, versioned datasets, consistent configs

### - doc-scribe
**Role**: Documentation and reproducibility assistant  
**Tools**: Markdown, GitHub Wiki, W&B Reports  
**Behavior**:
- Maintain README with setup, training, and inference instructions
- Auto-generate model cards from W&B runs and config metadata
- Ensure Roboflow dataset links and model IDs are documented
- Track changes to `config.yaml` and document rationale

### - mentor-bot
**Role**: Communication and feedback facilitator  
**Tools**: GitHub Issues, Discussions, Email Drafting  
**Behavior**:
- Draft follow-ups after demo sessions or PR merges
- Summarize feedback from reviewers and suggest next steps
- Track mentorship trial progress and flag missing responses
- Help onboard new contributors with Roboflow-specific guides

---

## 🔐 Permissions

| Agent        | Branch Access     | PR Review | Issue Commenting |
|--------------|-------------------|-----------|------------------|
| engineer     | `main`, `dev`     | ✅        | ✅               |
| doc-scribe   | `docs`, `main`    | ✅        | ✅               |
| mentor-bot   | `main`            | ❌        | ✅               |

---

## 📚 Context

Agents may read and reference:
- `README.md`, `config.yaml`, `train.py`, `inference.py`
- `rfdetr/`, `supervision/`, `notebooks/`
- W&B run metadata and Roboflow project/version strings

---

## 🧭 Mission Rules

- Never commit `.env` or API keys
- PRs touching `train.py`, `config.yaml`, or `inference.py` must be reviewed by `engineer`
- All training runs must log to W&B with project/run name matching the dataset
- Dataset usage must include version pinning (e.g., `project/version` in Roboflow)
- Inference scripts must use `get_model()` or `RFDETR*` classes from `rfdetr`

---

## 🧪 Protocols

### - Fine-Tuning Validation
- Confirm `ROBOFLOW_API_KEY` is loaded securely
- Validate dataset pull via `rf.load()` or CLI
- Ensure correct resolution and class count in `config.yaml`
- Check for `model.optimize_for_inference()` before export

### - Documentation Update
- Update README if CLI, config, or training logic changes
- Include example usage:
  ```bash
  python train.py --config config.yaml
  python inference.py --image path/to/image.jpg --model rf-model/1
  ```