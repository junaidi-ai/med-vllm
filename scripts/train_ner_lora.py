#!/usr/bin/env python3
"""
LoRA token-classification (NER) training script with optional BigBio aliases.

Examples:
  # Quick smoke on wikiann:en
  python scripts/train_ner_lora.py \
    --base-model dmis-lab/biobert-base-cased-v1.2 \
    --dataset wikiann:en \
    --epochs 1 --batch-size 8 --lr 2e-5 \
    --out outputs/ner-wikiann-smoke

  # BigBio BC5CDR (requires datasets<3.0 and 'bioc' Python package)
  python scripts/train_ner_lora.py \
    --base-model dmis-lab/biobert-base-cased-v1.2 \
    --dataset bc5cdr \
    --epochs 3 --batch-size 8 --lr 2e-5 \
    --out outputs/ner-bc5cdr

  # Push result to Hub umbrella repo as a PR
  env -u HF_TOKEN python scripts/train_ner_lora.py \
    --base-model dmis-lab/biobert-base-cased-v1.2 \
    --dataset wikiann:en --epochs 1 --batch-size 8 --lr 2e-5 \
    --out outputs/ner-wikiann-smoke \
    --push Junaidi-AI/med-vllm --path-in-repo checkpoints/ner-wikiann-smoke --create-pr
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any, Optional, List

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import set_seed
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score
from peft import LoraConfig, get_peft_model, TaskType

try:
    from huggingface_hub import HfApi
except Exception:  # optional
    HfApi = None  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA NER training")
    p.add_argument("--base-model", required=True)
    p.add_argument("--dataset", required=True, help="e.g., wikiann:en, conll2003, bc5cdr, ncbi_disease")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output dir")
    # Optional push-to-hub
    p.add_argument("--push", default=None, help="repo id to push to (umbrella)")
    p.add_argument("--path-in-repo", default=None)
    p.add_argument("--create-pr", action="store_true")
    return p.parse_args()


def resolve_dataset(dataset_name: str):
    """Load a token classification dataset supporting aliases and name:config.

    Returns (dataset, token_col, tag_col)
    """
    os.environ.setdefault("HF_DATASETS_DISABLE_LOCAL_IMPORTS", "1")
    ds_spec = (dataset_name or "").strip()

    # Aliases for BigBio (script-based; requires datasets<3.0 and 'bioc')
    alias_map = {
        "bc5cdr": ("bigbio/bc5cdr", "bigbio_ner"),
        "ncbi_disease": ("bigbio/ncbi_disease", "bigbio_ner"),
    }

    # Try loader
    if ds_spec.lower() in alias_map:
        name, config = alias_map[ds_spec.lower()]
        ds = load_dataset(name, config, trust_remote_code=True)
    else:
        if ":" in ds_spec:
            name, config = [s.strip() for s in ds_spec.split(":", 1)]
            ds = load_dataset(name, config)
        else:
            ds = load_dataset(ds_spec)

    if "train" not in ds:
        raise RuntimeError("Dataset must include a 'train' split")

    # Detect columns
    features = ds["train"].features
    token_candidates = ["tokens", "words"]
    tag_candidates = ["ner_tags", "tags", "labels", "ner_tags_general"]
    token_col = next((c for c in token_candidates if c in features), None)
    tag_col = next((c for c in tag_candidates if c in features), None)
    if not token_col or not tag_col:
        raise RuntimeError(
            f"Could not find token/tag columns in dataset. Looked for {token_candidates} and {tag_candidates}."
        )
    return ds, token_col, tag_col


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    ds, token_col, tag_col = resolve_dataset(args.dataset)

    # Label space
    label_list = ds["train"].features[tag_col].feature.names
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    # Some checkpoints (e.g., BioBERT) don't ship tokenizer files; fall back to bert-base-cased
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer from {args.base_model}: {e}\n"
              f"[WARN] Falling back to 'bert-base-cased' tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    base = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(base, peft_config)

    def tokenize_align(batch: Dict[str, Any]) -> Dict[str, Any]:
        # Produce labels aligned to subword tokenization
        new_labels: List[List[int]] = []
        encodings = tokenizer(
            batch[token_col], is_split_into_words=True, truncation=True, max_length=512, padding=True
        )
        for tokens, tags in zip(batch[token_col], batch[tag_col]):
            enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=512, padding=False)
            word_ids = enc.word_ids()
            lab: List[int] = []
            prev_wid: Optional[int] = None
            for wid in word_ids:
                if wid is None:
                    lab.append(-100)
                else:
                    tag_id = tags[wid]
                    if wid != prev_wid:
                        lab.append(tag_id)
                        prev_wid = wid
                    else:
                        lab.append(-100)
            new_labels.append(lab)
        encodings["labels"] = new_labels
        return encodings

    tokenized = ds.map(tokenize_align, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    metrics_out: Dict[str, float] = {}

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.argmax(-1)
        true_predictions = []
        true_labels = []
        for pred, lab in zip(preds, labels):
            curr_pred = []
            curr_lab = []
            for p_i, l_i in zip(pred, lab):
                if l_i != -100:
                    curr_pred.append(id2label[int(p_i)])
                    curr_lab.append(id2label[int(l_i)])
            true_predictions.append(curr_pred)
            true_labels.append(curr_lab)
        out = {
            "f1": f1_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
        metrics_out.update(out)
        return out

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation") or tokenized.get("dev") or tokenized.get("test"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print("=== Training complete ===")
    print("Metrics:", metrics_out)

    # Optional push-to-hub
    if args.push and HfApi is not None:
        api = HfApi()
        path_in_repo = args.path_in_repo or os.path.basename(args.out.rstrip("/"))
        print(f"Pushing to {args.push}:{path_in_repo} (create_pr={args.create_pr})")
        commit = api.upload_folder(
            repo_id=args.push,
            repo_type="model",
            folder_path=args.out,
            path_in_repo=path_in_repo,
            commit_message=f"Add NER LoRA checkpoint ({path_in_repo})",
            create_pr=args.create_pr,
        )
        print(commit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
