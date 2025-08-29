import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from medvllm.models.adapters import BioBERTAdapter, ClinicalBERTAdapter

try:
    from tests.medical.memory_profiler import MemoryProfiler  # type: ignore
except Exception:
    # Lightweight fallback profiler to avoid import-time failures outside test pkg
    class MemoryProfiler:  # type: ignore
        def __init__(self, device: str = "cpu"):
            self.device = device
            self.results = {}

        def profile(self):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield

            return _cm()


def benchmark_model(model, batch_size=8, seq_len=128, iterations=100, texts=None, device="cpu"):
    model.eval()
    device = next(model.parameters()).device if hasattr(model, "parameters") else device

    # Generate test input
    base_text = "This is a test sentence for benchmarking."
    input_text = texts or [base_text]
    if isinstance(input_text, list):
        if len(input_text) < batch_size:
            # pad/repeat to batch size
            input_text = (input_text * ((batch_size + len(input_text) - 1) // len(input_text)))[
                :batch_size
            ]
        else:
            input_text = input_text[:batch_size]
    else:
        input_text = [base_text] * batch_size

    # Warmup
    for _ in range(min(10, max(2, iterations // 10))):
        with torch.no_grad():
            inputs = model.preprocess_biomedical_text(input_text)
            _ = model(**inputs)

    # Benchmark
    start_time = time.time()
    mem_prof = MemoryProfiler(
        device=device.type if isinstance(device, torch.device) else str(device)
    )
    with torch.no_grad():
        for _ in tqdm(range(iterations), desc=f"Batch {batch_size}"):
            with mem_prof.profile():
                inputs = model.preprocess_biomedical_text(input_text)
                _ = model(**inputs)

    total_time = time.time() - start_time
    avg_latency = (total_time / iterations) * 1000  # ms
    tokens_per_sec = (batch_size * seq_len * iterations) / total_time

    return {
        "avg_latency_ms": avg_latency,
        "tokens_per_sec": tokens_per_sec,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "memory": getattr(mem_prof, "results", {}),
    }


def load_texts_from_jsonl(path: Path, max_lines: int = 128) -> list[str]:
    texts = []
    try:
        import json as _json

        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = _json.loads(line)
                t = obj.get("text")
                if isinstance(t, str) and t:
                    texts.append(t)
    except Exception:
        pass
    return texts


def parse_args():
    p = argparse.ArgumentParser(description="Run model/dataset benchmark sweep")
    p.add_argument(
        "--models",
        type=str,
        default="biobert,clinicalbert",
        help="Comma-separated model types: biobert,clinicalbert",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="benchmarks/datasets/mimic_notes_sample.jsonl,benchmarks/datasets/pubmed_sample.jsonl",
        help="Comma-separated JSONL files with a 'text' field",
    )
    p.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8], help="Batch sizes to test"
    )
    p.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[128, 256],
        help="Sequence lengths (for tokens/sec normalization only)",
    )
    p.add_argument("--iterations", type=int, default=50, help="Iterations per configuration")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output",
        type=str,
        default="benchmark_results/sweep_{}.json".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    print(f"Running benchmarks on {device}")

    model_map = {
        "biobert": (BioBERTAdapter, "monologg/biobert_v1.1_pubmed"),
        "clinicalbert": (ClinicalBERTAdapter, "emilyalsentzer/Bio_ClinicalBERT"),
    }

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [Path(d.strip()) for d in args.datasets.split(",") if d.strip()]

    os.makedirs(Path(args.output).parent, exist_ok=True)
    sweep_results = []

    for mtype in models:
        if mtype not in model_map:
            print(f"Skipping unknown model type: {mtype}")
            continue
        cls, default_id = model_map[mtype]
        print(f"\nLoading {mtype} -> {default_id}")
        adapter = cls.from_pretrained(default_id)
        adapter = adapter.to(device)
        if device.type == "cuda":
            adapter = adapter.half()

        for dpath in datasets:
            texts = (
                load_texts_from_jsonl(dpath, max_lines=max(args.batch_sizes))
                if dpath.exists()
                else None
            )
            for bs in args.batch_sizes:
                for sl in args.seq_lengths:
                    try:
                        res = benchmark_model(
                            adapter,
                            batch_size=bs,
                            seq_len=sl,
                            iterations=args.iterations,
                            texts=texts,
                            device=device,
                        )
                    except Exception as e:
                        print(f"Error benchmarking {mtype} on {dpath.name} (b{bs},s{sl}): {e}")
                        continue
                    record = {
                        "timestamp": datetime.now().isoformat(),
                        "model_type": mtype,
                        "model_id": default_id,
                        "dataset": str(dpath),
                        **res,
                        "device": str(device),
                    }
                    sweep_results.append(record)
                    print(
                        f"{mtype} {dpath.name} bs={bs} seq={sl}: {res['avg_latency_ms']:.2f} ms, {int(res['tokens_per_sec']):,} tok/s"
                    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"results": sweep_results}, f, indent=2)
    print(f"Saved sweep results to {args.output}")
