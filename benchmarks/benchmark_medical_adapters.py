"""Benchmark script for medical model adapters."""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from medvllm.models.adapters import BioBERTAdapter, ClinicalBERTAdapter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark medical model adapters")
    parser.add_argument(
        "--model-type",
        type=str,
        default="biobert",
        choices=["biobert", "clinicalbert"],
        help="Type of medical model to benchmark",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help=(
            "Hugging Face model identifier or local path. "
            "Defaults: biobert=monologg/biobert_v1.1_pubmed, clinicalbert=emilyalsentzer/Bio_ClinicalBERT"
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--use-optimized",
        action="store_true",
        help="Use optimized attention and layer implementations",
    )
    parser.add_argument(
        "--use-cuda-graphs",
        action="store_true",
        help="Use CUDA graphs for optimization",
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Use mixed precision (FP16) for inference",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results_cpu_smoke",
        help="Directory to save JSON results",
    )
    return parser.parse_args()


def generate_test_data(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Generate random test data."""
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def benchmark_adapter(
    adapter: torch.nn.Module,
    test_data: Dict[str, torch.Tensor],
    num_warmup: int,
    num_iterations: int,
    use_cuda_graphs: bool = False,
) -> Tuple[float, float]:
    """Benchmark the adapter.

    Args:
        adapter: The adapter to benchmark
        test_data: Test data dictionary
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        use_cuda_graphs: Whether to use CUDA graphs

    Returns:
        Tuple of (avg_latency_ms, tokens_per_second)
    """
    device = next(adapter.parameters()).device
    batch_size = test_data["input_ids"].shape[0]
    seq_length = test_data["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = adapter(**test_data)

    # Create CUDA graph if enabled
    if use_cuda_graphs and torch.cuda.is_available():
        # Create static inputs
        static_inputs = {k: v.clone() for k, v in test_data.items()}

        # Warmup for CUDA graph capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = adapter(**static_inputs)
        torch.cuda.current_stream().wait_stream(s)

        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_outputs = adapter(**static_inputs)

    # Benchmark
    latencies = []
    with torch.no_grad():
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold]Iter b{batch_size}s{seq_length}[/]"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
        ) as progress:
            t = progress.add_task("iter", total=num_iterations)
            for _ in range(num_iterations):
                if use_cuda_graphs and torch.cuda.is_available():
                    # Replay graph
                    start_time = time.time()
                    g.replay()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()
                else:
                    # Standard forward pass
                    start_time = time.time()
                    _ = adapter(**test_data)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()

                latencies.append((end_time - start_time) * 1000)  # Convert to ms
                progress.advance(t)

    # Calculate metrics
    avg_latency = np.mean(latencies)
    tokens_per_second = (batch_size * seq_length * 1000) / avg_latency

    return avg_latency, tokens_per_second


def main():
    """Main benchmarking function."""
    args = parse_args()

    # Set device
    device = torch.device(
        "cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Determine default model name if not provided
    default_names = {
        "biobert": "monologg/biobert_v1.1_pubmed",
        "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    }
    model_name = args.model_name or default_names[args.model_type]

    # Initialize adapter via from_pretrained
    if args.model_type == "biobert":
        adapter = BioBERTAdapter.from_pretrained(
            model_name,
            use_cuda_graphs=args.use_cuda_graphs,
            enable_mixed_precision=args.use_mixed_precision,
        )
    else:
        adapter = ClinicalBERTAdapter.from_pretrained(
            model_name,
            use_cuda_graphs=args.use_cuda_graphs,
            enable_mixed_precision=args.use_mixed_precision,
        )

    # Move to device and set to eval mode
    adapter = adapter.to(device)
    adapter.eval()

    # Enable mixed precision if specified
    if args.use_mixed_precision and torch.cuda.is_available():
        adapter = adapter.half()

    # Derive vocab size and max position embeddings
    if getattr(adapter, "tokenizer", None) is not None:
        vocab_size = len(adapter.tokenizer)
    else:
        vocab_size = getattr(adapter.model.config, "vocab_size", 30522)
    max_pos = getattr(adapter.model.config, "max_position_embeddings", 512)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Print model info
    num_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Model: {args.model_type.upper()}")
    print(f"Model name: {model_name}")
    print(f"Parameters: {num_params:,}")
    print(f"Optimized: {args.use_optimized}")
    print(f"CUDA Graphs: {args.use_cuda_graphs}")
    print(f"Mixed Precision: {args.use_mixed_precision}")
    print("-" * 80)

    # Benchmark different configurations
    results = []
    total_cfg = len(args.batch_sizes) * len(args.seq_lengths)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Benchmarking configs[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
    ) as progress:
        cfg_task = progress.add_task("cfgs", total=total_cfg)
        for batch_size in args.batch_sizes:
            for seq_length in args.seq_lengths:
                # Skip invalid combinations
                if seq_length > max_pos:
                    continue

                # Generate test data
                test_data = generate_test_data(
                    batch_size=batch_size,
                    seq_length=seq_length,
                    vocab_size=vocab_size,
                    device=device,
                )

                # Convert to half precision if using mixed precision
                if args.use_mixed_precision and torch.cuda.is_available():
                    test_data = {
                        k: v.half() if v.dtype == torch.float32 else v for k, v in test_data.items()
                    }

                # Benchmark
                avg_latency, tokens_per_second = benchmark_adapter(
                    adapter=adapter,
                    test_data=test_data,
                    num_warmup=args.num_warmup,
                    num_iterations=args.num_iterations,
                    use_cuda_graphs=args.use_cuda_graphs,
                )

                results.append(
                    {
                        "batch_size": batch_size,
                        "seq_length": seq_length,
                        "avg_latency_ms": avg_latency,
                        "tokens_per_second": tokens_per_second,
                    }
                )

                print(
                    f"Batch: {batch_size:3d}, "
                    f"Seq: {seq_length:4d}, "
                    f"Latency: {avg_latency:.2f} ms, "
                    f"Throughput: {tokens_per_second:,.0f} tokens/s"
                )
                progress.advance(cfg_task)

    # Print summary
    print("\n" + "=" * 80)
    print(f"{'BATCH':<6} | {'SEQ_LEN':<8} | {'LATENCY (ms)':<12} | {'THROUGHPUT (tokens/s)':<20}")
    print("-" * 80)

    for result in results:
        print(
            f"{result['batch_size']:<6} | "
            f"{result['seq_length']:<8} | "
            f"{result['avg_latency_ms']:>11.2f} | "
            f"{result['tokens_per_second']:>19,.0f}"
        )
    print("=" * 80)

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir,
        f"{args.model_type}_adapter_benchmark_{timestamp}.json",
    )
    payload = {
        "model_type": args.model_type,
        "model_name": model_name,
        "device": str(device),
        "num_warmup": args.num_warmup,
        "num_iterations": args.num_iterations,
        "use_cuda_graphs": args.use_cuda_graphs,
        "use_mixed_precision": args.use_mixed_precision,
        "batch_sizes": args.batch_sizes,
        "seq_lengths": args.seq_lengths,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_pos,
        "results": results,
        "timestamp": timestamp,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON results to: {output_path}")


if __name__ == "__main__":
    main()
