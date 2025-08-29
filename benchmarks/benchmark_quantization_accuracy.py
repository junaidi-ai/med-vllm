#!/usr/bin/env python3
"""
Minimal accuracy scaffold comparing baseline vs quantized models on a tiny dataset.

- Loads a HF sequence classification model and tokenizer
- Evaluates accuracy on a small CSV (fixture if available)
- Compares: baseline (no quant), dynamic int8 (CPU), and bnb-8bit (GPU if available)

Usage:
  python benchmarks/benchmark_quantization_accuracy.py \
    --model <hf_or_path> [--dataset-csv path.csv] [--limit 100]

Notes:
- If dataset or dependencies are missing, the script will skip gracefully.
- For bnb evaluation you need CUDA + bitsandbytes + compatible transformers.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as e:  # pragma: no cover
    print("transformers is required: pip install transformers", file=sys.stderr)
    raise


def load_dataset_csv(path: Optional[str], limit: int) -> Optional[List[Tuple[str, int]]]:
    # Try provided path; else fall back to project fixture
    if not path:
        fallback = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "tests",
            "fixtures",
            "data",
            "datasets",
            "text_classification_dataset.csv",
        )
        path = fallback

    if not os.path.exists(path):
        print(f"[info] Dataset CSV not found at {path}; skipping accuracy.")
        return None

    rows: List[Tuple[str, int]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "").strip().lower() != "test":
                    continue
                text = row.get("text")
                label = row.get("label")
                if text is None or label is None:
                    continue
                try:
                    y = int(label)
                except Exception:
                    # Only numeric labels supported by this scaffold
                    continue
                rows.append((text, y))
                if len(rows) >= limit:
                    break
    except Exception as e:
        print(f"[warn] Failed to read dataset: {e}")
        return None

    if not rows:
        print("[info] No usable rows found in dataset; skipping accuracy.")
        return None

    return rows


def predict_accuracy(
    model_name: str, bits: Optional[int], method: Optional[str], data: List[Tuple[str, int]]
) -> Optional[float]:
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"[warn] Failed to load tokenizer: {e}")
        return None

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}

    if bits == 8 and method and method.lower() in {"dynamic", "torch", "cpu"}:
        # Load fp32 model on CPU and apply dynamic quantization
        device = torch.device("cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs).to(
            device
        )
        model.eval()
        qapi = getattr(torch, "quantization", None)
        qfunc = getattr(qapi, "quantize_dynamic", None) if qapi is not None else None
        if callable(qfunc):
            model = qfunc(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)  # type: ignore
    elif bits == 8 and method and method.lower() == "bnb-8bit":
        if not torch.cuda.is_available():
            print("[info] CUDA not available; skipping bnb-8bit accuracy")
            return None
        load_kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs)
    elif bits is None and method is None:
        # Baseline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs).to(
            device
        )
        model.eval()
    else:
        print(f"[info] Unsupported config for this scaffold: bits={bits} method={method}")
        return None

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for text, y in data:
            enc = tok(text, return_tensors="pt", truncation=True, padding=False)
            # Move to model device if it has one
            device = next(model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            if not hasattr(out, "logits"):
                print("[warn] Model output has no logits; skipping.")
                return None
            pred = int(out.logits.argmax(dim=-1).item())
            correct += int(pred == y)
            total += 1

    return float(correct) / float(total) if total else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        help="HF repo id or local path for a sequence classification model",
    )
    ap.add_argument("--dataset-csv", type=str, default=None)
    ap.add_argument("--limit", type=int, default=64)
    args = ap.parse_args()

    data = load_dataset_csv(args.dataset_csv, limit=args.limit)
    if not data:
        print("{}")
        return

    results: List[Dict[str, Any]] = []

    # Baseline
    acc = predict_accuracy(args.model, None, None, data)
    results.append({"bits": None, "method": None, "accuracy": acc})

    # Dynamic int8
    acc = predict_accuracy(args.model, 8, "dynamic", data)
    results.append({"bits": 8, "method": "dynamic", "accuracy": acc})

    # bnb-8bit (if CUDA available)
    if torch.cuda.is_available():
        acc = predict_accuracy(args.model, 8, "bnb-8bit", data)
        results.append({"bits": 8, "method": "bnb-8bit", "accuracy": acc})

    # Print JSON summary
    import json

    print(json.dumps({"model": args.model, "results": results}, indent=2))


if __name__ == "__main__":
    main()
