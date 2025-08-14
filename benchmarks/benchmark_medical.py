"""Comprehensive benchmarking for medical model adapters."""

import argparse
import json
import os
import platform
import psutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

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
    test_accuracy: bool = True
    memory_profile: bool = True

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = DEFAULT_BATCH_SIZES
        if self.seq_lengths is None:
            self.seq_lengths = DEFAULT_SEQ_LENGTHS

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)


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
        # Generate random input IDs
        input_ids = torch.randint(
            low=100,
            high=self.model.config.vocab_size - 1,
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

    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result to file."""
        timestamp = int(time.time())
        filename = f"{self.config.model_type}_benchmark_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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
    )

    benchmark = MedicalModelBenchmark(config)
    benchmark.run_benchmark()
