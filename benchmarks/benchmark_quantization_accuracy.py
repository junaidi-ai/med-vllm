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

# Validation utilities (optional runtime use)
from medvllm.eval.validation import (
    compute_classification_metrics,
    mcnemar_test_equivalence,
    threshold_check,
)
from medvllm.eval.thresholds import DEFAULT_THRESHOLDS, load_thresholds_from_file
from datetime import datetime

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
    label_map: Dict[str, int] = {}
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
                # Allow numeric or string labels; map strings to ints deterministically
                y: Optional[int] = None
                try:
                    y = int(label)
                except Exception:
                    key = str(label)
                    if key not in label_map:
                        label_map[key] = len(label_map)
                    y = label_map[key]
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
    model_name: str,
    bits: Optional[int],
    method: Optional[str],
    data: List[Tuple[str, int]],
    *,
    return_preds: bool = False,
    return_probs: bool = False,
) -> Optional[float] | Tuple[Optional[float], List[int], List[int], Optional[List[float]]]:
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
        try:
            import bitsandbytes as _bnb  # type: ignore  # noqa: F401
        except Exception:
            print("[info] bitsandbytes not installed; skipping bnb-8bit accuracy")
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
    y_true_list: List[int] = []
    y_pred_list: List[int] = []
    prob_list: List[float] = []

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
            logits = out.logits
            pred = int(logits.argmax(dim=-1).item())
            correct += int(pred == y)
            total += 1
            if return_preds:
                y_true_list.append(int(y))
                y_pred_list.append(int(pred))
                if return_probs:
                    # Use probability for class 1 if binary; else max prob as a generic score
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    if probs.numel() >= 2:
                        prob_list.append(float(probs[1].item()))
                    else:
                        prob_list.append(float(probs.max().item()))

    acc = float(correct) / float(total) if total else None
    if return_preds:
        return acc, y_true_list, y_pred_list, (prob_list if return_probs else None)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        help="HF repo id or local path for a sequence classification model",
    )
    ap.add_argument("--dataset-csv", type=str, default=None)
    ap.add_argument("--limit", type=int, default=64)
    # Validation/report options
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Run statistical validation vs baseline and write report",
    )
    ap.add_argument(
        "--out-dir", type=str, default="reports", help="Output directory for validation report"
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="quant_validation",
        help="Output filename prefix for validation artifacts",
    )
    ap.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Threshold in key=value form (repeatable), e.g., accuracy=0.95",
    )
    ap.add_argument(
        "--thresholds-file",
        type=str,
        default=None,
        help="Load thresholds from JSON/YAML file (flat or namespaced)",
    )
    ap.add_argument(
        "--thresholds-preset",
        type=str,
        default=None,
        choices=sorted(list(DEFAULT_THRESHOLDS.keys())),
        help="Use a built-in thresholds preset (e.g., classification.general)",
    )
    ap.add_argument(
        "--save-probs-json",
        action="store_true",
        help="Save y_true and per-model probabilities JSON for ROC/PR curves",
    )
    ap.add_argument(
        "--signoff",
        action="append",
        default=[],
        help="Sign-off metadata in key=value form (repeatable)",
    )
    ap.add_argument(
        "--signoff-file", type=str, default=None, help="Load sign-off metadata from JSON/YAML file"
    )
    ap.add_argument(
        "--fail-on-thresholds",
        action="store_true",
        help="Exit non-zero if optimized model fails thresholds",
    )
    args = ap.parse_args()

    data = load_dataset_csv(args.dataset_csv, limit=args.limit)
    if not data:
        print("{}")
        return

    results: List[Dict[str, Any]] = []

    # Baseline
    base_out = predict_accuracy(
        args.model, None, None, data, return_preds=args.validate, return_probs=args.save_probs_json
    )
    if args.validate:
        base_acc, y_true, base_preds, base_probs = base_out  # type: ignore[assignment]
    else:
        base_acc = base_out  # type: ignore[assignment]
        y_true, base_preds, base_probs = [], [], None
    results.append({"bits": None, "method": None, "accuracy": base_acc})

    # Dynamic int8
    dyn_out = predict_accuracy(
        args.model,
        8,
        "dynamic",
        data,
        return_preds=args.validate,
        return_probs=args.save_probs_json,
    )
    if args.validate:
        dyn_acc, _, dyn_preds, dyn_probs = dyn_out  # type: ignore[assignment]
    else:
        dyn_acc = dyn_out  # type: ignore[assignment]
        dyn_preds, dyn_probs = [], None
    results.append({"bits": 8, "method": "dynamic", "accuracy": dyn_acc})

    # bnb-8bit (if CUDA available)
    if torch.cuda.is_available():
        bnb_out = predict_accuracy(
            args.model,
            8,
            "bnb-8bit",
            data,
            return_preds=args.validate,
            return_probs=args.save_probs_json,
        )
        if args.validate:
            if isinstance(bnb_out, tuple):
                bnb_acc, _, bnb_preds, bnb_probs = bnb_out  # type: ignore[assignment]
            else:
                bnb_acc = bnb_out  # type: ignore[assignment]
                bnb_preds, bnb_probs = [], None
        else:
            bnb_acc = bnb_out  # type: ignore[assignment]
            bnb_preds, bnb_probs = [], None
        results.append({"bits": 8, "method": "bnb-8bit", "accuracy": bnb_acc})

    # Print JSON summary
    import json

    summary = {"model": args.model, "results": results}
    print(json.dumps(summary, indent=2))

    # Optional validation: compute metrics/McNemar vs baseline and write JSON report
    if args.validate:
        from pathlib import Path

        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

        # Compose thresholds
        th_map: Dict[str, float] = {}
        if args.thresholds_preset:
            th_map.update(DEFAULT_THRESHOLDS.get(args.thresholds_preset, {}))
        if args.thresholds_file:
            try:
                loaded = load_thresholds_from_file(args.thresholds_file)
                if isinstance(loaded, dict):
                    if (
                        args.thresholds_preset
                        and args.thresholds_preset in loaded
                        and isinstance(loaded[args.thresholds_preset], dict)
                    ):
                        th_map.update(
                            {
                                k: float(v)
                                for k, v in loaded[args.thresholds_preset].items()
                                if isinstance(v, (int, float))
                            }
                        )
                    else:
                        th_map.update(
                            {k: float(v) for k, v in loaded.items() if isinstance(v, (int, float))}
                        )
            except Exception:
                pass
        for item in args.threshold or []:
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    th_map[k.strip()] = float(v.strip())
                except Exception:
                    pass

        # Metrics for baseline
        base_metrics = (
            compute_classification_metrics(y_true, base_preds) if y_true and base_preds else None
        )

        validations: List[Dict[str, Any]] = []
        probs_payload: Dict[str, Any] = {}
        probs_path: Optional[str] = None
        if args.save_probs_json and y_true:
            probs_payload["y_true"] = y_true
        # Dynamic int8 vs baseline
        if y_true and base_preds and dyn_preds:
            dyn_metrics = compute_classification_metrics(y_true, dyn_preds)
            mc = mcnemar_test_equivalence(y_true, base_preds, dyn_preds)
            if (
                args.save_probs_json
                and 'dyn_probs' in locals()
                and dyn_probs is not None
                and base_probs is not None
            ):
                probs_payload.setdefault("scores_a", base_probs)
                probs_payload["scores_b_dynamic"] = dyn_probs
            validations.append(
                {
                    "pair": "baseline_vs_dynamic8",
                    "baseline_metrics": base_metrics.__dict__ if base_metrics else None,
                    "optimized_metrics": dyn_metrics.__dict__,
                    "mcnemar": mc.__dict__,
                    "thresholds": th_map,
                    "threshold_results": {
                        "baseline": threshold_check(base_metrics, th_map)
                        if base_metrics and th_map
                        else {},
                        "optimized": threshold_check(dyn_metrics, th_map) if th_map else {},
                    },
                }
            )

        # bnb-8bit vs baseline
        if y_true and base_preds and 'bnb_out' in locals() and bnb_preds:
            bnb_metrics = compute_classification_metrics(y_true, bnb_preds)
            mc = mcnemar_test_equivalence(y_true, base_preds, bnb_preds)
            if (
                args.save_probs_json
                and 'bnb_probs' in locals()
                and bnb_probs is not None
                and base_probs is not None
            ):
                probs_payload.setdefault("scores_a", base_probs)
                probs_payload["scores_b_bnb"] = bnb_probs
            validations.append(
                {
                    "pair": "baseline_vs_bnb8",
                    "baseline_metrics": base_metrics.__dict__ if base_metrics else None,
                    "optimized_metrics": bnb_metrics.__dict__,
                    "mcnemar": mc.__dict__,
                    "thresholds": th_map,
                    "threshold_results": {
                        "baseline": threshold_check(base_metrics, th_map)
                        if base_metrics and th_map
                        else {},
                        "optimized": threshold_check(bnb_metrics, th_map) if th_map else {},
                    },
                }
            )

        # Sign-off metadata
        signoff_map: Dict[str, Any] = {}
        if args.signoff_file:
            try:
                with open(args.signoff_file, "r", encoding="utf-8") as f:
                    text = f.read()
                try:
                    signoff_map.update(json.loads(text))
                except Exception:
                    try:
                        import yaml  # type: ignore

                        data = yaml.safe_load(text)
                        if isinstance(data, dict):
                            signoff_map.update(data)
                    except Exception:
                        pass
            except Exception:
                pass
        for item in args.signoff or []:
            if "=" in item:
                k, v = item.split("=", 1)
                vv: Any = v.strip()
                if isinstance(vv, str) and vv.lower() in {"true", "false"}:
                    vv = vv.lower() == "true"
                signoff_map[k.strip()] = vv
        signoff_map.setdefault("date", datetime.utcnow().isoformat() + "Z")

        report = {
            "model": args.model,
            "dataset_csv": args.dataset_csv,
            "limit": args.limit,
            "comparisons": validations,
            "signoff": signoff_map,
        }
        out_path = os.path.join(args.out_dir, f"{args.prefix}_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[info] Validation report written to {out_path}")

        if args.save_probs_json and probs_payload:
            probs_path = os.path.join(args.out_dir, f"{args.prefix}_scores.json")
            with open(probs_path, "w", encoding="utf-8") as f:
                json.dump(probs_payload, f)
            print(f"[info] ROC/PR scores written to {probs_path}")

        # Optional CI gate: fail if any optimized comparison fails thresholds
        if args.fail_on_thresholds and th_map:

            def any_fail(vs: List[Dict[str, Any]]) -> bool:
                for item in vs:
                    opt = item.get("threshold_results", {}).get("optimized", {})
                    if opt and not all(bool(v) for v in opt.values()):
                        return True
                return False

            if any_fail(validations):
                raise SystemExit(1)


if __name__ == "__main__":
    main()
