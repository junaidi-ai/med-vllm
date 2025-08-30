#!/usr/bin/env python
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    # Both benchmarks print their JSON payload to stdout; capture it
    stdout = proc.stdout.strip()
    try:
        data = json.loads(stdout)
    except Exception:
        data = {"raw_stdout": stdout}
    return data


def main():
    parser = argparse.ArgumentParser(description="Run unified-profiler-enabled benchmarks and aggregate a profiling report.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--trace-dir", default="benchmarks/profiles")
    parser.add_argument("--out", default=None, help="Output JSON path (default: benchmarks/reports/profiling_report_<ts>.json)")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else (repo / "benchmarks" / "reports" / f"profiling_report_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare per-benchmark output paths under a temp dir
    tmp_dir = repo / "benchmarks" / "benchmark_results_cpu_smoke"
    tmp_dir.mkdir(exist_ok=True)

    # Attention benchmark (small)
    attn_out = tmp_dir / f"attn_{args.device}_{timestamp}.json"
    attn_cmd = [
        sys.executable,
        "benchmarks/benchmark_attention.py",
        "--device", args.device,
        "--emit-trace",
        "--trace-dir", str(trace_dir),
        "--out", str(attn_out),
        "--seq", "128",
        "--heads", "4",
        "--iters", "2",
        "--warmup", "1",
        "--attn-softmaxv-bench",
    ]

    # Imaging benchmark (small)
    img_out = tmp_dir / f"imaging_{args.device}_{timestamp}.json"
    img_cmd = [
        sys.executable,
        "benchmarks/benchmark_imaging.py",
        "--device", args.device,
        "--dtype", "fp32" if args.device == "cpu" else "bf16",
        "--conv-type", "2d",
        "--in-ch", "8",
        "--batch", "2",
        "--batches", "2",
        "--emit-trace",
        "--trace-dir", str(trace_dir),
        "--out", str(img_out),
        "--depthwise-bench",
        "--depthwise-bench-iters", "20",
        "--depthwise-bench-sizes", "8x128x128,32x256x256",
    ]

    results = {}

    try:
        results["attention"] = run(attn_cmd, cwd=str(repo))
    except subprocess.CalledProcessError as e:
        results["attention"] = {"error": e.stderr.strip()}

    try:
        results["imaging"] = run(img_cmd, cwd=str(repo))
    except subprocess.CalledProcessError as e:
        results["imaging"] = {"error": e.stderr.strip()}

    # Derive a simple per-op summary from Chrome traces (if any exist)
    op_summary = {}
    try:
        trace_files = sorted([p for p in trace_dir.glob("*.json") if p.is_file()])
        for tf in trace_files:
            with open(tf, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    continue
            events = data if isinstance(data, list) else data.get("traceEvents", [])
            for ev in events:
                # Consider complete events (ph == 'X') with a name and duration
                if isinstance(ev, dict) and ev.get("ph") == "X":
                    name = ev.get("name", "unknown")
                    dur_us = ev.get("dur", 0)
                    if name not in op_summary:
                        op_summary[name] = 0
                    op_summary[name] += int(dur_us)
    except Exception:
        op_summary = {}

    # Top ops by total duration (microseconds)
    if op_summary:
        top_ops = sorted(op_summary.items(), key=lambda x: x[1], reverse=True)[:15]
        results["op_summary_us_top15"] = [{"op": k, "total_us": v} for k, v in top_ops]

    # Include metadata
    results["meta"] = {
        "device": args.device,
        "trace_dir": str(Path(args.trace_dir).resolve()),
        "timestamp": datetime.now().isoformat(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({"report": str(out_path), "trace_dir": str(trace_dir)}, indent=2))


if __name__ == "__main__":
    main()
