from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

from medvllm.tasks import NERProcessor
from medvllm.utils.ner_metrics import compute_ner_strict_metrics


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_benchmark(dataset_path: Path, warmup: int = 2, runs: int = 5) -> Dict:
    rows = _load_jsonl(dataset_path)
    test_rows = [r for r in rows if r.get("split") == "test"] or rows

    from types import SimpleNamespace

    cfg = SimpleNamespace(ner_enable_extended_gazetteer=True)
    proc = NERProcessor(inference_pipeline=None, config=cfg)

    # Prepare gold
    gold_docs = [
        [
            {"start": int(e["start"]), "end": int(e["end"]), "type": str(e["type"]).lower()}
            for e in r.get("entities", [])
        ]
        for r in test_rows
    ]

    # Warmup
    for _ in range(warmup):
        for r in test_rows:
            _ = proc.extract_entities(r["text"])

    # Timed runs
    latencies = []
    preds_last = None
    for _ in range(runs):
        start = time.perf_counter()
        pred_docs = []
        for r in test_rows:
            res = proc.extract_entities(r["text"])
            pred_docs.append(
                [
                    {"start": int(e["start"]), "end": int(e["end"]), "type": str(e["type"]).lower()}
                    for e in res.entities
                ]
            )
        end = time.perf_counter()
        latencies.append(end - start)
        preds_last = pred_docs

    assert preds_last is not None
    metrics = compute_ner_strict_metrics(gold_docs, preds_last)

    num_examples = len(test_rows)
    avg_latency = mean(latencies)
    throughput = num_examples / avg_latency if avg_latency > 0 else 0.0

    return {
        "num_examples": num_examples,
        "avg_run_latency_sec": avg_latency,
        "throughput_examples_per_sec": throughput,
        "runs": runs,
        "latencies_sec": latencies,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark NER extraction and strict metrics")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tests/fixtures/data/datasets/ner_dataset.jsonl",
        help="Path to JSONL dataset with fields: text, entities, split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results_cpu_smoke/ner_benchmark_results.json",
        help="Where to save the benchmark JSON results",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(dataset_path, warmup=args.warmup, runs=args.runs)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print concise summary
    micro = results["metrics"]["micro"]
    print(
        json.dumps(
            {
                "num_examples": results["num_examples"],
                "throughput_examples_per_sec": results["throughput_examples_per_sec"],
                "micro_f1": micro["f1"],
            }
        )
    )


if __name__ == "__main__":
    main()
