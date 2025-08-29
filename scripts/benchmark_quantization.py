#!/usr/bin/env python3
"""
Quick benchmark comparing baseline vs 8-bit/4-bit quantization.

Usage:
  python scripts/benchmark_quantization.py --model <hf_or_path> \
      [--bits 8|4] [--method dynamic|bnb-8bit|bnb-nf4] [--iters 10] \
      [--csv out.csv] [--charts-dir charts/]

If --bits/--method are omitted, runs a baseline (no quantization) first,
then attempts 8-bit dynamic (CPU) and bnb-8bit (GPU if available).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import torch

from medvllm.optim.medical_optimizer import MedicalModelOptimizer, OptimizerConfig

try:
    from transformers import AutoModelForCausalLM
except Exception as e:  # pragma: no cover
    print("transformers is required: pip install transformers", file=sys.stderr)
    raise


def load_model(name_or_path: str) -> Any:
    return AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True)


def run_case(model_name: str, bits: int | None, method: str | None, iters: int) -> Dict[str, Any]:
    model = load_model(model_name)
    cfg = OptimizerConfig(model_name_or_path=model_name, quantization_bits=bits, quantization_method=method)
    opt = MedicalModelOptimizer(model, cfg)
    if bits:
        opt.quantize(bits=bits, method=method or "dynamic")
    opt.optimize_memory()
    results = opt.benchmark(["Patient presents with fever and cough.", "CT reveals pulmonary nodules."], iterations=iters)
    return {
        "bits": bits,
        "method": method,
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, choices=[4, 8], default=None)
    ap.add_argument("--method", type=str, default=None)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--csv", type=str, default=None, help="Write results as CSV")
    ap.add_argument(
        "--charts-dir",
        type=str,
        default=None,
        help="Directory to save charts (PNG). Requires matplotlib.",
    )
    args = ap.parse_args()

    cases: List[Dict[str, Any]] = []
    if args.bits is None and args.method is None:
        # Baseline
        cases.append(run_case(args.model, None, None, args.iters))
        # Dynamic int8
        cases.append(run_case(args.model, 8, "dynamic", args.iters))
        # bnb 8-bit if CUDA
        if torch.cuda.is_available():
            cases.append(run_case(args.model, 8, "bnb-8bit", args.iters))
    else:
        cases.append(run_case(args.model, args.bits, args.method, args.iters))

    summary = {"model": args.model, "benchmarks": cases}
    print(json.dumps(summary, indent=2))

    # CSV export (no pandas dependency required)
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model",
                    "bits",
                    "method",
                    "batch_size",
                    "elapsed_s",
                    "iterations",
                    "tokens_processed",
                    "throughput_tok_per_s",
                    "cuda_allocated_mb",
                    "cuda_reserved_mb",
                ]
            )
            for c in cases:
                bits = c.get("bits")
                method = c.get("method")
                res: Dict[str, Any] = c.get("results", {})
                for bs, metrics in res.items():
                    writer.writerow(
                        [
                            args.model,
                            bits,
                            method,
                            bs,
                            metrics.get("elapsed_s"),
                            metrics.get("iterations"),
                            metrics.get("tokens_processed"),
                            metrics.get("throughput_tok_per_s"),
                            metrics.get("cuda_allocated_mb"),
                            metrics.get("cuda_reserved_mb"),
                        ]
                    )

    # Charts (optional)
    if args.charts_dir:
        try:
            import os
            import matplotlib.pyplot as plt  # type: ignore

            os.makedirs(args.charts_dir, exist_ok=True)

            # Throughput chart per case
            for c in cases:
                method_label = f"bits={c.get('bits')} method={c.get('method')}"
                res: Dict[str, Any] = c.get("results", {})
                batches = sorted(res.keys(), key=lambda x: int(x))
                thr = [res[b].get("throughput_tok_per_s") or 0.0 for b in batches]
                elp = [res[b].get("elapsed_s") or 0.0 for b in batches]

                plt.figure()
                plt.title(f"Throughput - {method_label}")
                plt.plot([int(b) for b in batches], thr, marker="o")
                plt.xlabel("Batch size")
                plt.ylabel("Throughput (tok/s)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(args.charts_dir, f"throughput_{c.get('bits')}_{c.get('method')}.png"))
                plt.close()

                plt.figure()
                plt.title(f"Elapsed - {method_label}")
                plt.plot([int(b) for b in batches], elp, marker="o", color="orange")
                plt.xlabel("Batch size")
                plt.ylabel("Elapsed (s) total for iterations")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(args.charts_dir, f"elapsed_{c.get('bits')}_{c.get('method')}.png"))
                plt.close()

                # CUDA memory charts if metrics present
                if any(res[b].get("cuda_allocated_mb") for b in batches):
                    mem = [res[b].get("cuda_allocated_mb") or 0 for b in batches]
                    plt.figure()
                    plt.title(f"CUDA Allocated MB - {method_label}")
                    plt.plot([int(b) for b in batches], mem, marker="o", color="green")
                    plt.xlabel("Batch size")
                    plt.ylabel("MB")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.charts_dir, f"cuda_mem_{c.get('bits')}_{c.get('method')}.png"))
                    plt.close()
        except Exception as e:
            print(f"[warn] Skipped charts due to: {e}")


if __name__ == "__main__":
    main()
