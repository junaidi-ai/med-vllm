"""Comprehensive benchmarking for medical model adapters."""

import argparse
import json
import os
import platform
import csv
from collections import Counter
import psutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from medvllm.models.adapters import BioBERTAdapter, ClinicalBERTAdapter
from tests.medical.memory_profiler import MemoryProfiler


def get_system_info() -> Dict[str, Any]:
    """Collect system and hardware information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        },
    }

    if torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "device_count": torch.cuda.device_count(),
        }

    return info


# Default benchmark parameters
DEFAULT_BATCH_SIZES = [1, 4, 8, 16]
DEFAULT_SEQ_LENGTHS = [128, 256, 512, 1024]
DEFAULT_NUM_ITERATIONS = 10
DEFAULT_WARMUP_ITERS = 2


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs."""

    model_type: str = "biobert"  # or "clinicalbert"
    batch_sizes: List[int] = None
    seq_lengths: List[int] = None
    num_iterations: int = DEFAULT_NUM_ITERATIONS
    warmup_iterations: int = DEFAULT_WARMUP_ITERS
    use_kv_cache: bool = True
    precision: str = "fp16"  # "fp32" or "fp16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "benchmark_results"
    test_accuracy: bool = False
    dataset_csv: Optional[str] = None
    memory_profile: bool = True
    debug_io: bool = False
    # Optional Hugging Face dataset evaluation (majority baseline on small slice)
    hf_dataset: Optional[str] = None
    hf_subset: Optional[str] = None
    hf_split: Optional[str] = "train[:100]"
    hf_label_column: Optional[str] = None

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = DEFAULT_BATCH_SIZES
        if self.seq_lengths is None:
            self.seq_lengths = DEFAULT_SEQ_LENGTHS

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Default dataset path for fixture-based classification metrics
        if self.test_accuracy and not self.dataset_csv:
            default_csv = (
                Path(__file__).parent.parent
                / "tests/fixtures/data/datasets/text_classification_dataset.csv"
            )
            if default_csv.exists():
                self.dataset_csv = str(default_csv)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    model_type: str
    batch_size: int
    seq_length: int
    use_kv_cache: bool
    precision: str
    device: str

    # Performance metrics
    avg_latency_ms: float
    tokens_per_second: float
    memory_usage_mb: Dict[str, float]

    # System info
    timestamp: float = None
    gpu_info: Dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

        if self.gpu_info is None and torch.cuda.is_available():
            self.gpu_info = {
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
            }

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "BenchmarkResult":
        """Create from dictionary."""
        return cls(**data)


class MedicalModelBenchmark:
    """Benchmarking harness for medical model adapters."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark."""
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._load_model()

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Load the specified model."""
        model_class = BioBERTAdapter if self.config.model_type == "biobert" else ClinicalBERTAdapter
        model_name = (
            "monologg/biobert_v1.1_pubmed"
            if self.config.model_type == "biobert"
            else "emilyalsentzer/Bio_ClinicalBERT"
        )

        print(f"Loading {self.config.model_type}...")
        model = model_class.from_pretrained(model_name)

        # Set precision
        if self.config.precision == "fp16" and self.device.type == "cuda":
            model = model.half()

        # Move to device
        model = model.to(self.device)

        # Configure KV cache
        model.enable_cache() if self.config.use_kv_cache else model.disable_cache()

        return model

    def generate_inputs(self, batch_size: int, seq_length: int) -> Dict:
        """Generate dummy inputs for benchmarking."""
        # Determine vocab size robustly across adapters
        vocab_size = self._get_vocab_size()

        # Generate random input IDs
        input_ids = torch.randint(
            low=100,
            high=max(101, vocab_size - 1),
            size=(batch_size, seq_length),
            device=self.device,
            dtype=torch.long,
        )

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _get_vocab_size(self) -> int:
        """Best-effort retrieval of vocab size from adapter/tokenizer/model."""
        # Prefer tokenizer length if available (adapters may extend embeddings to match)
        try:
            if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                return len(self.model.tokenizer)
        except Exception:
            pass

        # Try underlying HF model config via common attribute path "model.config"
        try:
            hf_cfg = getattr(getattr(self.model, "model", None), "config", None)
            if hf_cfg is not None and hasattr(hf_cfg, "vocab_size"):
                return int(hf_cfg.vocab_size)
        except Exception:
            pass

        # Fallback to a safe BERT-like default
        print("Warning: Could not determine vocab size from adapter; defaulting to 30522")
        return 30522

    def run_benchmark(self) -> List[BenchmarkResult]:
        """Run the benchmark with the current configuration."""
        results = []

        for batch_size in self.config.batch_sizes:
            for seq_length in self.config.seq_lengths:
                print(f"\nBenchmarking - Batch: {batch_size}, SeqLen: {seq_length}")

                # Warmup
                for _ in range(self.config.warmup_iterations):
                    inputs = self.generate_inputs(batch_size, seq_length)
                    with torch.no_grad():
                        _ = self.model(**inputs)

                # Benchmark
                latencies = []
                mem_profiler = MemoryProfiler(device=self.device.type)

                for _ in range(self.config.num_iterations):
                    inputs = self.generate_inputs(batch_size, seq_length)

                    with mem_profiler.profile():
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            _ = self.model(**inputs)
                        end_time = time.perf_counter()

                    latencies.append((end_time - start_time) * 1000)  # ms

                # Calculate metrics
                avg_latency = np.mean(latencies)
                tokens_per_second = (batch_size * seq_length) / (avg_latency / 1000)

                # Create result
                result = BenchmarkResult(
                    model_type=self.config.model_type,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    use_kv_cache=self.config.use_kv_cache,
                    precision=self.config.precision,
                    device=str(self.device),
                    avg_latency_ms=avg_latency,
                    tokens_per_second=tokens_per_second,
                    memory_usage_mb=mem_profiler.results,
                )

                results.append(result)
                self._save_result(result)
                self._print_result(result)

        return results

    # --- Classification accuracy on fixtures (majority baseline) ---
    def evaluate_fixture_classification(self) -> Optional[Dict[str, Any]]:
        """Evaluate simple majority-class baseline accuracy on a fixture CSV.

        Returns a dict with metrics and dataset info, or None if unavailable.
        """
        if not self.config.test_accuracy:
            return None

        csv_path = self.config.dataset_csv
        if not csv_path or not os.path.exists(csv_path):
            print(f"[classification] Skipping: dataset CSV not found at {csv_path}")
            return None

        # Load dataset
        train_labels, test_labels = [], []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    split = str(row.get("split", "")).strip().lower()
                    label = row.get("label")
                    if not label:
                        continue
                    if split == "train":
                        train_labels.append(label)
                    elif split == "test":
                        test_labels.append(label)
        except Exception as e:
            print(f"[classification] Failed to read CSV: {e}")
            return None

        if not test_labels:
            print("[classification] Skipping: no test labels found in dataset")
            return None

        # Majority label from train (fallback to overall if train empty)
        source_for_majority = train_labels if train_labels else (train_labels + test_labels)
        majority_label = Counter(source_for_majority).most_common(1)[0][0]
        y_true = test_labels
        y_pred = [majority_label] * len(test_labels)

        # Compute metrics using our utility (requires sklearn)
        try:
            from medvllm.utils.metrics import compute_classification_metrics

            metrics = compute_classification_metrics(y_true, y_pred, average="macro")
        except Exception as e:
            print(f"[classification] Skipping metrics (dependency missing or error): {e}")
            return None

        result = {
            "model_type": self.config.model_type,
            "dataset_csv": csv_path,
            "baseline": "majority_class",
            "majority_label": majority_label,
            "num_train": len(train_labels),
            "num_test": len(test_labels),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        # Save and print
        self._save_classification_result(result)
        self._print_classification_result(result)
        return result

    def evaluate_hf_dataset_classification(self) -> Optional[Dict[str, Any]]:
        """Optionally evaluate a small slice from a HF dataset via majority baseline.

        Uses a user-provided dataset name/subset/split and label column. If the
        datasets library is missing or the dataset cannot be loaded (e.g., no
        internet and not cached), this will print a message and return None.
        """
        if not self.config.hf_dataset:
            return None

        # Lazy import to avoid hard dependency
        try:
            from datasets import load_dataset
        except Exception as e:
            print(f"[classification][hf] Skipping: datasets lib unavailable: {e}")
            return None

        name = self.config.hf_dataset
        subset = self.config.hf_subset
        split = self.config.hf_split or "train[:100]"

        try:
            ds = (
                load_dataset(name, subset, split=split)
                if subset
                else load_dataset(name, split=split)
            )
        except Exception as e:
            print(
                f"[classification][hf] Failed to load dataset {name} (subset={subset}, split={split}): {e}"
            )
            return None

        # Determine label column
        candidate_cols = [c for c in [self.config.hf_label_column, "label", "final_decision"] if c]
        label_col = None
        for c in candidate_cols:
            if c in ds.column_names:
                label_col = c
                break
        if label_col is None:
            print(
                f"[classification][hf] No suitable label column found in dataset columns: {ds.column_names}"
            )
            return None

        labels = list(ds[label_col])
        if not labels:
            print("[classification][hf] No labels found in loaded split; skipping")
            return None

        majority_label = Counter(labels).most_common(1)[0][0]
        y_true = labels
        y_pred = [majority_label] * len(labels)

        try:
            from medvllm.utils.metrics import compute_classification_metrics

            metrics = compute_classification_metrics(y_true, y_pred, average="macro")
        except Exception as e:
            print(f"[classification][hf] Skipping metrics (dependency missing or error): {e}")
            return None

        result = {
            "model_type": self.config.model_type,
            "dataset": name,
            "subset": subset,
            "split": split,
            "label_column": label_col,
            "baseline": "majority_class",
            "majority_label": majority_label,
            "num_examples": len(labels),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        # Save alongside other classification outputs
        self._save_classification_result(result)
        self._print_classification_result(result)
        return result

    def _save_classification_result(self, result: Dict[str, Any]):
        ts = int(time.time())
        filename = f"{self.config.model_type}_classification_metrics_{ts}.json"
        filepath = self.output_dir / filename
        if self.config.debug_io:
            print(f"[debug] Writing classification metrics JSON to: {filepath}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            # Always surface failures
            print(f"[debug] Failed to write classification JSON: {e}")
            return
        if self.config.debug_io:
            try:
                exists = filepath.exists()
                size = filepath.stat().st_size if exists else -1
                print(f"[debug] Classification JSON write complete. exists={exists} size={size}")
            except Exception as e:
                print(f"[debug] Post-write check failed for classification JSON: {e}")

    @staticmethod
    def _print_classification_result(result: Dict[str, Any]):
        print("\n" + "=" * 50)
        print(f"Classification Metrics - {result['model_type']} (baseline: {result['baseline']})")
        print("=" * 50)
        print(f"Dataset:         {result['dataset_csv']}")
        print(f"Majority Label:  {result['majority_label']}")
        print(f"Train/Test:      {result['num_train']}/{result['num_test']}")
        print("-" * 50)
        m = result["metrics"]
        print(f"Accuracy:        {m['accuracy']:.4f}")
        print(f"Precision:       {m['precision']:.4f}")
        print(f"Recall:          {m['recall']:.4f}")
        print(f"F1:              {m['f1']:.4f}")
        print("=" * 50 + "\n")

    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result to file."""
        timestamp = int(time.time())
        filename = f"{self.config.model_type}_benchmark_{timestamp}.json"
        filepath = self.output_dir / filename

        if self.config.debug_io:
            print(f"[debug] Writing benchmark JSON to: {filepath}")
        try:
            with open(filepath, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception as e:
            # Always surface failures
            print(f"[debug] Failed to write benchmark JSON: {e}")
            return
        if self.config.debug_io:
            try:
                exists = filepath.exists()
                size = filepath.stat().st_size if exists else -1
                print(f"[debug] Benchmark JSON write complete. exists={exists} size={size}")
            except Exception as e:
                print(f"[debug] Post-write check failed for benchmark JSON: {e}")

    @staticmethod
    def _print_result(result: BenchmarkResult):
        """Print benchmark result in a readable format."""
        print("\n" + "=" * 50)
        print(f"Benchmark Results - {result.model_type}")
        print("=" * 50)
        print(f"Batch Size:      {result.batch_size}")
        print(f"Sequence Length: {result.seq_length}")
        print(f"KV Cache:        {'Enabled' if result.use_kv_cache else 'Disabled'}")
        print(f"Precision:       {result.precision.upper()}")
        print(f"Device:          {result.device}")
        print("-" * 50)
        print(f"Avg Latency:     {result.avg_latency_ms:.2f} ms")
        print(f"Throughput:      {result.tokens_per_second:,.0f} tokens/sec")

        if "gpu_allocated_delta_mb" in result.memory_usage_mb:
            print("\nGPU Memory Usage:")
            print(f"  Allocated: {result.memory_usage_mb['gpu_allocated_delta_mb']:.2f} MB")
            print(
                f"  Cached:    {result.memory_usage_mb['gpu_cached_end_mb'] - result.memory_usage_mb['gpu_cached_start_mb']:.2f} MB"
            )

        print("=" * 50 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark medical model adapters")
    parser.add_argument(
        "--model",
        type=str,
        choices=["biobert", "clinicalbert"],
        default="biobert",
        help="Model type to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENGTHS,
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help="Number of iterations per benchmark",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERS,
        help="Number of warmup iterations",
    )
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable KV cache")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16"],
        default="fp16",
        help="Precision to use (fp32 or fp16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--test-accuracy",
        action="store_true",
        help="Evaluate simple classification metrics on the fixture dataset",
    )
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=None,
        help="Path to a CSV dataset with columns: text,label,source,split",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Hugging Face dataset name (e.g., pubmed_qa, bigbio/med_qa) for opt-in small-sample eval",
    )
    parser.add_argument(
        "--hf-subset",
        type=str,
        default=None,
        help="Optional dataset subset/config name (e.g., pqa_labeled for pubmed_qa)",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train[:100]",
        help="Split slice for the HF dataset (e.g., train[:100])",
    )
    parser.add_argument(
        "--hf-label-column",
        type=str,
        default=None,
        help="Label column in the HF dataset (auto-detects 'label' or 'final_decision' if not provided)",
    )
    parser.add_argument(
        "--debug-io",
        action="store_true",
        help="Print debug information when writing JSON outputs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # GPU fallback: if CUDA requested but not available, fall back to CPU
    if str(args.device).lower().startswith("cuda") and not torch.cuda.is_available():
        print(
            "[info] CUDA not available; falling back to CPU (set --device cpu explicitly to silence this message)"
        )
        args.device = "cpu"

    config = BenchmarkConfig(
        model_type=args.model,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
        use_kv_cache=not args.no_kv_cache,
        precision=args.precision,
        device=args.device,
        output_dir=args.output_dir,
        test_accuracy=args.test_accuracy,
        dataset_csv=args.dataset_csv,
        debug_io=args.debug_io,
        hf_dataset=args.hf_dataset,
        hf_subset=args.hf_subset,
        hf_split=args.hf_split,
        hf_label_column=args.hf_label_column,
    )

    benchmark = MedicalModelBenchmark(config)
    benchmark.run_benchmark()
    # Optional classification metrics on fixtures
    benchmark.evaluate_fixture_classification()
    # Optional classification metrics on a small Hugging Face dataset sample
    benchmark.evaluate_hf_dataset_classification()
