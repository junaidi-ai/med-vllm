"""Generate performance reports from benchmark results.

Supports two formats:
1) Per-run JSON objects (e.g., benchmark_medical.py outputs) containing metrics directly.
2) Sweep JSON files with a top-level {"results": [...]} list (from run_benchmarks.py).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def load_benchmark_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load and normalize benchmark result files from a directory.

    Accepts files like benchmark_*.json and sweep_*.json, but will consider any .json.
    """
    records: List[Dict[str, Any]] = []
    results_dir = Path(results_dir)

    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            print(f"Warning: Could not parse {result_file}")
            continue

        # Sweep format: {"results": [ ... ]}
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            for item in data["results"]:
                if isinstance(item, dict):
                    rec = normalize_record(item)
                    rec["_source_file"] = str(result_file)
                    records.append(rec)
            continue

        # Single-record format (benchmark_medical.py saves per-config files)
        if isinstance(data, dict):
            rec = normalize_record(data)
            rec["_source_file"] = str(result_file)
            records.append(rec)

    return records


def normalize_record(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize varying schema keys into a common structure for reporting."""
    out = dict(d)  # shallow copy
    # Unify sequence length
    if "seq_length" not in out and "seq_len" in out:
        out["seq_length"] = out.get("seq_len")
    # Unify tokens/sec
    if "tokens_per_second" not in out and "tokens_per_sec" in out:
        out["tokens_per_second"] = out.get("tokens_per_sec")
    # Unify memory structure
    if "memory_usage_mb" not in out and "memory" in out and isinstance(out["memory"], dict):
        out["memory_usage_mb"] = out["memory"]
    # Ensure model_type is present
    out.setdefault("model_type", d.get("model", d.get("model_type", "unknown")))
    return out


def generate_markdown_report(results: List[Dict[str, Any]], output_file: str):
    """Generate a markdown report from benchmark results."""
    if not results:
        print("No benchmark results found.")
        return

    # Try to extract any system info if present
    system_info = {}
    for r in results:
        si = r.get("system_info")
        if isinstance(si, dict):
            system_info = si
            break

    # Group results by (model_type, dataset)
    results_by_group: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        key = f"{result.get('model_type', 'unknown')} | {Path(str(result.get('dataset',''))).name or '-'}"
        results_by_group.setdefault(key, []).append(result)

    # Generate markdown
    markdown = [
        "# Medical vLLM Benchmark Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## System Information",
        "```json",
        json.dumps(system_info, indent=2),
        "```\n",
        "## Benchmark Results\n",
    ]

    # Add results table per group
    for group_key, model_results in results_by_group.items():
        markdown.extend(
            [
                f"### {group_key}\n",
                "| Batch | Seq | Avg Latency (ms) | Tokens/sec | Device | Model ID | GPU Mem Δ (MB) |",
                "|------:|----:|-----------------:|-----------:|--------|----------|---------------:|",
            ]
        )

        for result in sorted(
            model_results, key=lambda r: (r.get("batch_size", 0), r.get("seq_length", 0))
        ):
            mem = result.get("memory_usage_mb", {})
            gpu_delta = None
            if isinstance(mem, dict):
                # Prefer explicit delta if present, else try allocated_end-start
                gpu_delta = mem.get("gpu_allocated_delta_mb")
                if gpu_delta is None and all(
                    k in mem for k in ("gpu_allocated_end_mb", "gpu_allocated_start_mb")
                ):
                    try:
                        gpu_delta = float(mem["gpu_allocated_end_mb"]) - float(
                            mem["gpu_allocated_start_mb"]
                        )
                    except Exception:
                        gpu_delta = None

            markdown.append(
                f"| {result.get('batch_size', 'N/A')} "
                f"| {result.get('seq_length', 'N/A')} "
                f"| {float(result.get('avg_latency_ms', 0.0)):.2f} "
                f"| {float(result.get('tokens_per_second', 0.0)):.2f} "
                f"| {result.get('device', result.get('device_type', '-'))} "
                f"| {result.get('model_id', result.get('model_name','-'))} "
                f"| {gpu_delta if gpu_delta is not None else '-'} |"
            )

        markdown.append("\n")

    # Write report
    # Aggregates
    markdown.append("## Aggregates (mean tokens/sec per group)\n")
    markdown.extend(
        [
            "| Group | Mean tokens/sec | Count |",
            "|-------|-----------------:|------:|",
        ]
    )
    for group_key, model_results in results_by_group.items():
        vals = [
            float(r.get("tokens_per_second", 0.0))
            for r in model_results
            if r.get("tokens_per_second") is not None
        ]
        if vals:
            mean_val = sum(vals) / max(1, len(vals))
            markdown.append(f"| {group_key} | {mean_val:.2f} | {len(vals)} |")
    markdown.append("")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown))

    print(f"Report generated: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_report.md", help="Output markdown file"
    )

    args = parser.parse_args()

    results = load_benchmark_results(args.results_dir)
    generate_markdown_report(results, args.output)
