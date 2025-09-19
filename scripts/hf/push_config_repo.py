#!/usr/bin/env python3
"""
Export MedicalModelConfig examples and push a config-only repository to the Hugging Face Hub.

Usage:
  python scripts/hf/push_config_repo.py \
      --repo-id Junaidi-AI/med-vllm-config \
      [--private] [--no-create] [--export-dir ./.hf_export/med-vllm-config]

Prereqs:
  - pip install huggingface_hub
  - huggingface-cli login

This script creates several example configurations:
  - NER default
  - Classification with labels
  - Generation default
It writes both JSON and YAML for each example and uploads the folder to the Hub.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from huggingface_hub import create_repo, upload_folder

from medvllm.medical.config.models.medical_config import MedicalModelConfig


README_MD = """---
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

## Usage (Python)

```python
from medvllm.medical.config.models.medical_config import MedicalModelConfig
cfg = MedicalModelConfig.from_pretrained("Junaidi-AI/med-vllm")
print(cfg.task_type)
```

Or directly load a specific example folder if exported as a subfolder with
its own config files.

## Medical Disclaimer

This repository and associated configurations are provided for research and
engineering purposes only. They are not intended for clinical decision-making.
Always involve qualified healthcare professionals and ensure compliance with
applicable regulations (e.g., HIPAA, GDPR). Avoid using PHI/PII.

## License

MIT
"""


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save(cfg: MedicalModelConfig, out_dir: Path, stem: str) -> None:
    _ensure_dir(out_dir)
    # Write JSON and YAML side-by-side
    cfg.to_json(out_dir / f"{stem}.json")
    cfg.to_yaml(out_dir / f"{stem}.yaml")


def build_examples(export_dir: Path) -> None:
    # Top-level README
    (export_dir / "README.md").write_text(README_MD, encoding="utf-8")

    # NER example
    ner_cfg = MedicalModelConfig(
        model="dmis-lab/biobert-base-cased-v1.2",
        task_type="ner",
        medical_entity_types=["disease", "drug", "procedure"],
        ner_confidence_threshold=0.85,
    )
    _save(ner_cfg, export_dir / "examples" / "ner", stem="config")

    # Classification example
    clf_cfg = MedicalModelConfig(
        model="emilyalsentzer/Bio_ClinicalBERT",
        task_type="classification",
        classification_labels=["diagnosis", "treatment", "follow-up"],
        batch_size=16,
    )
    _save(clf_cfg, export_dir / "examples" / "classification", stem="config")

    # Generation example (keep minimal)
    gen_cfg = MedicalModelConfig(
        model="gpt2",
        task_type="generation",
        max_medical_seq_length=1024,
    )
    _save(gen_cfg, export_dir / "examples" / "generation", stem="config")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Junaidi-AI/med-vllm", help="HF repo id")
    ap.add_argument("--private", action="store_true", help="Create as private repo")
    ap.add_argument(
        "--no-create",
        action="store_true",
        help="Do not attempt to create the remote repo (must already exist)",
    )
    ap.add_argument(
        "--export-dir",
        default=".hf_export/med-vllm-config",
        help="Local export directory",
    )
    ap.add_argument(
        "--create-pr",
        action="store_true",
        help="If set, push changes via a PR instead of direct commit (useful for org repos requiring PRs)",
    )
    args = ap.parse_args()

    export_dir = Path(args.export_dir)
    _ensure_dir(export_dir)

    # Build local folder contents
    build_examples(export_dir)

    # Create repo if requested
    if not args.no_create:
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Upload folder
    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(export_dir),
        repo_type="model",
        create_pr=args.create_pr,
        commit_message="Initial config export (README, examples)",
    )
    print(f"Uploaded {export_dir} to {args.repo_id}")


if __name__ == "__main__":
    main()
