---
license: mit
language: en
tags:
  - medical
  - config
  - med-vllm
library_name: medvllm
pipeline_tag: token-classification
---

# Med vLLM (Config-first Repository)

This repository serves as a config-first landing for the Med vLLM stack.

It contains example configuration files and is intended to help users discover
and consume the `MedicalModelConfig` from the Hub via `from_pretrained`, and to
use these as starting points for training or inference in medical NLP tasks.

## Contents

- NER config example (`examples/ner/`)
- Classification config example (`examples/classification/`)
- Generation config example (`examples/generation/`)

## Install

```bash
pip install medvllm
```

## Quickstart (Python)

```python
from medvllm.medical.config.models.medical_config import MedicalModelConfig
cfg = MedicalModelConfig.from_pretrained("Junaidi-AI/med-vllm")
print(cfg.task_type)
```

Or directly load a specific example folder if exported as a subfolder with
its own config files.

## Examples

- NER: [`examples/ner/config.json`](./examples/ner/config.json) | [`examples/ner/config.yaml`](./examples/ner/config.yaml)
- Classification: [`examples/classification/config.json`](./examples/classification/config.json) | [`examples/classification/config.yaml`](./examples/classification/config.yaml)
- Generation: [`examples/generation/config.json`](./examples/generation/config.json) | [`examples/generation/config.yaml`](./examples/generation/config.yaml)

Use these as starting points and customize fields like `task_type`, `classification_labels`, `medical_entity_types`, and domain settings.

## Tasks supported

- Named Entity Recognition (NER)
- Text Classification
- Text Generation

All tasks share a unified configuration schema via `MedicalModelConfig`.

## Weights roadmap

This repo currently focuses on configs. Model weights/adapters will be added progressively:

- BioBERT/ClinicalBERT adapters
- Task-specific fine-tuned checkpoints (NER/Classification)

Follow the repo for updates or open a Discussion to request specific checkpoints.

## Debug and logging

By default, verbose config debug prints are silenced. To enable them for troubleshooting, set:

```bash
export MEDVLLM_CONFIG_DEBUG=1
```

## Medical Disclaimer

This repository and associated configurations are provided for research and
engineering purposes only. They are not intended for clinical decision-making.
Always involve qualified healthcare professionals and ensure compliance with
applicable regulations (e.g., HIPAA, GDPR). Avoid using PHI/PII.

## License

MIT
