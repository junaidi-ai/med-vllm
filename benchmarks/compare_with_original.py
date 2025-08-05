"""Benchmark comparing Med-vLLM with original implementations."""

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from medvllm.models import MedicalModel
from benchmarks.benchmark_utils import (
    get_system_info,
    measure_memory,
    format_size,
    log_benchmark_result
)

class OriginalVsMedVLLMComparison:
    """Compare performance between original models and Med-vLLM implementations."""
    
    def __init__(self, model_name: str = "biobert", device: str = None):
        """Initialize the comparison runner.
        
        Args:
            model_name: Name of the model to benchmark (e.g., 'biobert', 'clinicalbert')
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
    def load_models(self):
        """Load both original and Med-vLLM models."""
        # Load Med-vLLM model
        self.medvllm_model = MedicalModel.from_pretrained(
            f"medical-{self.model_name}",
            device=self.device
        )
        
        # Load original model (mock for now, would be replaced with actual loading)
        self.original_model = self._load_original_model()
    
    def _load_original_model(self):
        """Load the original model implementation."""
        # This is a placeholder - in a real implementation, this would load
        # the original model (e.g., from Hugging Face)
        return {
            "name": f"original-{self.model_name}",
            "predict": lambda x: [{"dummy": "result"} for _ in x]
        }
    
    def generate_test_data(self, num_samples: int = 100, max_length: int = 128) -> List[str]:
        """Generate test data for benchmarking."""
        # In a real implementation, this would load actual medical text samples
        return [
            "Patient presents with chest pain and shortness of breath." * 2,
            "History of diabetes and hypertension. Current medications include metformin and lisinopril." * 2,
            "CT scan shows no evidence of pulmonary embolism. Mild cardiomegaly noted." * 2,
        ] * (num_samples // 3 + 1)[:num_samples]
    
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
            for _ in range(num_runs):
                start = time.time()
                self.medvllm_model.predict(batch)
                medvllm_times.append(time.time() - start)
            
            # Benchmark original
            original_times = []
            for _ in range(num_runs):
                start = time.time()
                self.original_model["predict"](batch)
                original_times.append(time.time() - start)
            
            # Calculate stats
            medvllm_avg = np.mean(medvllm_times[1:])  # Skip first run (warmup)
            original_avg = np.mean(original_times[1:])
            
            self.results.append({
                "batch_size": batch_size,
                "medvllm_avg_time": medvllm_avg,
                "original_avg_time": original_avg,
                "speedup": original_avg / medvllm_avg if medvllm_avg > 0 else 0,
                "memory_usage": measure_memory()
            })
    
    def print_results(self):
        """Print benchmark results in a readable format."""
        print("\n" + "="*80)
        print(f"Benchmark Results: {self.model_name}")
        print("="*80)
        print(f"Device: {self.device}")
        print("-"*80)
        
        headers = ["Batch", "Med-vLLM (s)", "Original (s)", "Speedup", "Memory"]
        row_format = "{:>8} {:>12} {:>12} {:>10} {:>12}"
        print(row_format.format(*headers))
        print("-" * 60)
        
        for result in self.results:
            print(row_format.format(
                result["batch_size"],
                f"{result['medvllm_avg_time']:.4f}",
                f"{result['original_avg_time']:.4f}",
                f"{result['speedup']:.2f}x",
                format_size(result["memory_usage"])
            ))
        
        print("="*80 + "\n")
    
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
            "timestamp": time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"Results saved to {output_file}")


def main():
    """Run the benchmark comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Med-vLLM with original implementations')
    parser.add_argument('--model', type=str, default='biobert',
                       help='Model to benchmark (default: biobert)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16],
                       help='Batch sizes to test')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs per configuration')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Directory to save benchmark results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Run the benchmark
    benchmark = OriginalVsMedVLLMComparison(
        model_name=args.model,
        device=args.device
    )
    
    print(f"Loading models...")
    benchmark.load_models()
    
    print(f"Running benchmarks with batch sizes: {args.batch_sizes}")
    benchmark.run_benchmark(
        batch_sizes=args.batch_sizes,
        num_runs=args.runs
    )
    
    # Print and save results
    benchmark.print_results()
    benchmark.save_results(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
