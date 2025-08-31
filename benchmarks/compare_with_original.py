"""Benchmark comparing Med-vLLM with original implementations."""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch

# Add project root to path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from medvllm.models import MedicalModel
from benchmarks.benchmark_utils import get_system_info, measure_memory, format_size
from medvllm.eval.validation import (
    compute_classification_metrics,
    mcnemar_test_equivalence,
    threshold_check,
)
from medvllm.eval.thresholds import DEFAULT_THRESHOLDS, load_thresholds_from_file


class OriginalVsMedVLLMComparison:
    """Compare performance between original models and Med-vLLM implementations."""

    def __init__(self, model_name: str = "biobert", device: str = None):
        """Initialize the comparison runner.

        Args:
            model_name: Name of the model to benchmark (e.g., 'biobert', 'clinicalbert')
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []

    def load_models(self):
        """Load both original and Med-vLLM models."""
        # Load Med-vLLM model
        self.medvllm_model = MedicalModel.from_pretrained(
            f"medical-{self.model_name}", device=self.device
        )

        # Load original model (mock for now, would be replaced with actual loading)
        self.original_model = self._load_original_model()

    def _load_original_model(self):
        """Load the original model implementation."""
        # This is a placeholder - in a real implementation, this would load
        # the original model (e.g., from Hugging Face)
        return {
            "name": f"original-{self.model_name}",
            "predict": lambda x: [{"dummy": "result"} for _ in x],
        }

    def generate_test_data(self, num_samples: int = 100, max_length: int = 128) -> List[str]:
        """Generate test data for benchmarking."""
        # In a real implementation, this would load actual medical text samples
        base = [
            "Patient presents with chest pain and shortness of breath." * 2,
            "History of diabetes and hypertension. Current medications include metformin and lisinopril."
            * 2,
            "CT scan shows no evidence of pulmonary embolism. Mild cardiomegaly noted." * 2,
        ]
        return (base * (num_samples // len(base) + 1))[:num_samples]

    def run_benchmark(self, batch_sizes: List[int] = None, num_runs: int = 5):
        """Run the benchmark comparison.

        Args:
            batch_sizes: List of batch sizes to test
            num_runs: Number of runs per configuration
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16]

        test_data = self.generate_test_data(max(batch_sizes) * 2)

        for batch_size in batch_sizes:
            batch = test_data[:batch_size]

            # Benchmark Med-vLLM
            medvllm_times = []
            med_predict = getattr(self.medvllm_model, "predict", None)
            if callable(med_predict):
                for _ in range(num_runs):
                    start = time.time()
                    med_predict(batch)
                    medvllm_times.append(time.time() - start)
            else:
                # No predict API available; record zeros and continue
                medvllm_times = [0.0 for _ in range(num_runs)]

            # Benchmark original
            original_times = []
            for _ in range(num_runs):
                start = time.time()
                self.original_model["predict"](batch)
                original_times.append(time.time() - start)

            # Calculate stats
            medvllm_avg = (
                np.mean(medvllm_times[1:])
                if len(medvllm_times) > 1
                else (medvllm_times[0] if medvllm_times else 0.0)
            )
            original_avg = np.mean(original_times[1:])

            self.results.append(
                {
                    "batch_size": batch_size,
                    "medvllm_avg_time": medvllm_avg,
                    "original_avg_time": original_avg,
                    "speedup": original_avg / medvllm_avg if medvllm_avg > 0 else 0,
                    "memory_usage": measure_memory(),
                    "notes": "medvllm_model.predict not available"
                    if not callable(med_predict)
                    else "",
                }
            )

    def print_results(self):
        """Print benchmark results in a readable format."""
        print("\n" + "=" * 80)
        print(f"Benchmark Results: {self.model_name}")
        print("=" * 80)
        print(f"Device: {self.device}")
        print("-" * 80)

        headers = ["Batch", "Med-vLLM (s)", "Original (s)", "Speedup", "Memory"]
        row_format = "{:>8} {:>12} {:>12} {:>10} {:>12}"
        print(row_format.format(*headers))
        print("-" * 60)

        for result in self.results:
            mem = result["memory_usage"]
            if isinstance(mem, dict):
                rss = mem.get("rss_mb")
                cuda = mem.get("cuda_allocated_mb")
                if rss is not None and cuda is not None:
                    mem_str = f"RSS {rss:.1f}MB, CUDA {cuda:.1f}MB"
                elif rss is not None:
                    mem_str = f"RSS {rss:.1f}MB"
                else:
                    mem_str = "-"
            else:
                from benchmarks.benchmark_utils import format_size

                mem_str = format_size(float(mem)) if isinstance(mem, (int, float)) else "-"
            print(
                row_format.format(
                    result["batch_size"],
                    f"{result['medvllm_avg_time']:.4f}",
                    f"{result['original_avg_time']:.4f}",
                    f"{result['speedup']:.2f}x",
                    mem_str,
                )
            )

        print("=" * 80 + "\n")

    def save_results(self, output_dir: str = "benchmark_results"):
        """Save benchmark results to a JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"comparison_{self.model_name}_{timestamp}.json"

        result_data = {
            "system_info": get_system_info(),
            "model": self.model_name,
            "device": self.device,
            "results": self.results,
            "timestamp": time.time(),
        }

        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"Results saved to {output_file}")

        return output_file


def main():
    """Run the benchmark comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare Med-vLLM with original implementations")
    parser.add_argument(
        "--model",
        type=str,
        default="biobert",
        help="Model to benchmark (default: biobert)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Batch sizes to test",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per configuration")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda or cpu)")
    # Optional accuracy validation inputs (external CSV with columns)
    parser.add_argument(
        "--validation-csv",
        type=str,
        default=None,
        help="CSV with true/pred columns to run accuracy validation",
    )
    parser.add_argument("--y-true-col", type=str, default=None)
    parser.add_argument(
        "--y-pred-a-col", type=str, default=None, help="Column for original/baseline predictions"
    )
    parser.add_argument(
        "--y-pred-b-col", type=str, default=None, help="Column for Med-vLLM/optimized predictions"
    )
    parser.add_argument(
        "--threshold", action="append", default=[], help="Threshold key=value; repeatable"
    )
    parser.add_argument(
        "--thresholds-file", type=str, default=None, help="Load thresholds from JSON/YAML file"
    )
    parser.add_argument(
        "--thresholds-preset",
        type=str,
        default=None,
        choices=sorted(list(DEFAULT_THRESHOLDS.keys())),
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="compare_validation",
        help="Filename prefix for validation report",
    )

    args = parser.parse_args()

    # Run the benchmark
    benchmark = OriginalVsMedVLLMComparison(model_name=args.model, device=args.device)

    print("Loading models...")
    benchmark.load_models()

    print(f"Running benchmarks with batch sizes: {args.batch_sizes}")
    benchmark.run_benchmark(batch_sizes=args.batch_sizes, num_runs=args.runs)

    # Print and save results
    benchmark.print_results()
    out_json = benchmark.save_results(output_dir=args.output_dir)

    # Optional validation branch if CSV provided
    if args.validation_csv and args.y_true_col and args.y_pred_a_col and args.y_pred_b_col:
        import csv as _csv

        def _read_csv_rows(path: str, y_col: str, a_col: str, b_col: str):
            y_true: List[str] = []
            a_pred: List[str] = []
            b_pred: List[str] = []
            with open(path, "r", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    if y_col in row and a_col in row and b_col in row:
                        y_true.append(str(row[y_col]))
                        a_pred.append(str(row[a_col]))
                        b_pred.append(str(row[b_col]))
            return y_true, a_pred, b_pred

        y_true, a_pred, b_pred = _read_csv_rows(
            args.validation_csv, args.y_true_col, args.y_pred_a_col, args.y_pred_b_col
        )

        # Compute metrics and McNemar
        m_a = compute_classification_metrics(y_true, a_pred)
        m_b = compute_classification_metrics(y_true, b_pred)
        mc = mcnemar_test_equivalence(y_true, a_pred, b_pred)

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

        report = {
            "benchmark_results_json": str(out_json),
            "validation_csv": args.validation_csv,
            "columns": {
                "y_true": args.y_true_col,
                "y_pred_a": args.y_pred_a_col,
                "y_pred_b": args.y_pred_b_col,
            },
            "metrics": {
                "baseline": m_a.__dict__,
                "optimized": m_b.__dict__,
            },
            "mcnemar": mc.__dict__,
            "thresholds": th_map,
            "threshold_results": {
                "baseline": threshold_check(m_a, th_map) if th_map else {},
                "optimized": threshold_check(m_b, th_map) if th_map else {},
            },
        }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.report_prefix}_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Validation report written to {out_path}")


if __name__ == "__main__":
    main()
