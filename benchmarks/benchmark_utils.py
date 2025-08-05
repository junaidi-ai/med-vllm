"""Utility functions for benchmarking and testing."""

import json
import platform
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def get_system_info() -> Dict[str, Any]:
    """Collect system and hardware information.
    
    Returns:
        Dictionary containing system information including CPU, GPU, memory, etc.
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(),
            "cpu_freq": psutil.cpu_freq()._asdict() if hasattr(psutil, 'cpu_freq') else {},
        },
        "memory": psutil.virtual_memory()._asdict(),
    }
    
    # Add GPU information if available
    if torch.cuda.is_available():
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            gpu_info[f"cuda:{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(i)),
                "memory_allocated": torch.cuda.memory_allocated(i) / (1024 ** 2),  # MB
                "memory_reserved": torch.cuda.memory_reserved(i) / (1024 ** 2),    # MB
                "memory_cached": torch.cuda.memory_reserved(i) / (1024 ** 2),      # MB
            }
        info["gpu"] = gpu_info
    
    return info


def measure_memory() -> Dict[str, float]:
    """Measure current memory usage.
    
    Returns:
        Dictionary with memory usage statistics in MB.
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    
    memory_stats = {
        'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
    }
    
    if torch.cuda.is_available():
        memory_stats.update({
            'cuda_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'cuda_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'cuda_max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
            'cuda_max_reserved_mb': torch.cuda.max_memory_reserved() / (1024 * 1024),
        })
    
    return memory_stats


def format_size(size_bytes: float) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}PB"


def log_benchmark_result(result: Dict[str, Any], output_dir: Union[str, Path] = "benchmark_results"):
    """Save benchmark result to a JSON file.
    
    Args:
        result: Dictionary containing benchmark results
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = result.get("model", "unknown").replace("/", "-")
    
    output_file = output_dir / f"benchmark_{model_name}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return output_file


def time_function(func, *args, warmup: int = 3, iterations: int = 10, **kwargs) -> Dict[str, float]:
    """Time a function with warmup and multiple iterations.
    
    Args:
        func: Function to time
        *args: Positional arguments to pass to the function
        warmup: Number of warmup iterations
        iterations: Number of timing iterations
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Dictionary with timing statistics in seconds
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    
    # Timing
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate statistics
    times_np = np.array(times)
    return {
        'min': float(np.min(times_np)),
        'max': float(np.max(times_np)),
        'mean': float(np.mean(times_np)),
        'median': float(np.median(times_np)),
        'std': float(np.std(times_np)),
        'total': float(np.sum(times_np)),
        'iterations': iterations,
    }


def generate_test_sequences(
    vocab_size: int = 50000,
    min_length: int = 10,
    max_length: int = 512,
    num_sequences: int = 1000,
    seed: int = 42
) -> List[List[int]]:
    """Generate random test sequences for benchmarking.
    
    Args:
        vocab_size: Size of the vocabulary
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        num_sequences: Number of sequences to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of token ID sequences
    """
    rng = np.random.RandomState(seed)
    
    sequences = []
    for _ in range(num_sequences):
        length = rng.randint(min_length, max_length + 1)
        seq = rng.randint(0, vocab_size, size=length).tolist()
        sequences.append(seq)
    
    return sequences


def check_gpu_memory() -> Tuple[bool, str]:
    """Check if GPU is available and has sufficient memory.
    
    Returns:
        Tuple of (is_available, message)
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    
    try:
        # Try allocating a small tensor to check if GPU is accessible
        x = torch.randn(1, device='cuda')
        del x
        return True, "GPU is available and accessible"
    except Exception as e:
        return False, f"GPU error: {str(e)}"
