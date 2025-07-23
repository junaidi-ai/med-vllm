"""Memory profiling utilities for medical model testing."""

import time
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import psutil
import torch


class MemoryProfiler:
    """Memory profiler for tracking memory usage during model inference."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the memory profiler.
        
        Args:
            device: The device to monitor ("cuda" or "cpu").
        """
        self.device = device
        self.start_mem: Optional[Dict[str, float]] = None
        self.peak_mem: Dict[str, float] = {}
        self.start_time: float = 0
        self.end_time: float = 0
        
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        stats = {
            'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            stats.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            })
            
        return stats
    
    def start(self) -> None:
        """Start memory profiling."""
        self.start_mem = self._get_memory_stats()
        self.start_time = time.time()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def stop(self) -> Dict[str, float]:
        """Stop memory profiling and return results.
        
        Returns:
            Dictionary containing memory usage statistics.
        """
        if self.start_mem is None:
            raise RuntimeError("Profiling not started. Call start() first.")
            
        end_mem = self._get_memory_stats()
        self.end_time = time.time()
        
        # Calculate memory deltas
        results = {
            'time_elapsed': self.end_time - self.start_time,
            'start_rss_mb': self.start_mem['rss_mb'],
            'end_rss_mb': end_mem['rss_mb'],
            'rss_delta_mb': end_mem['rss_mb'] - self.start_mem['rss_mb'],
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            results.update({
                'gpu_allocated_start_mb': self.start_mem['gpu_allocated_mb'],
                'gpu_allocated_end_mb': end_mem['gpu_allocated_mb'],
                'gpu_allocated_delta_mb': end_mem['gpu_allocated_mb'] - self.start_mem['gpu_allocated_mb'],
                'gpu_cached_start_mb': self.start_mem['gpu_cached_mb'],
                'gpu_cached_end_mb': end_mem['gpu_cached_mb'],
                'gpu_peak_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
                'gpu_peak_cached_mb': torch.cuda.max_memory_reserved() / (1024 * 1024),
            })
        
        return results
    
    @contextmanager
    def profile(self):
        """Context manager for memory profiling.
        
        Example:
            with MemoryProfiler().profile() as prof:
                # Code to profile
                model(inputs)
            print(prof.results)
        """
        self.start()
        try:
            yield self
        finally:
            self.results = self.stop()
            
    def get_summary(self) -> str:
        """Get a formatted summary of memory usage."""
        if not hasattr(self, 'results'):
            return "No profiling data available. Call stop() after start()."
            
        r = self.results
        summary = [
            "\n=== Memory Profiling Summary ===",
            f"Time Elapsed: {r['time_elapsed']:.4f} seconds",
            "\nCPU Memory:",
            f"  RSS: {r['end_rss_mb']:.2f} MB (Δ {r['rss_delta_mb']:+.2f} MB)",
        ]
        
        if 'gpu_allocated_delta_mb' in r:
            summary.extend([
                "\nGPU Memory:",
                f"  Allocated: {r['gpu_allocated_end_mb']:.2f} MB (Δ {r['gpu_allocated_delta_mb']:+.2f} MB)",
                f"  Cached: {r['gpu_cached_end_mb']:.2f} MB (Δ {r['gpu_cached_end_mb'] - r['gpu_cached_start_mb']:+.2f} MB)",
                f"  Peak Allocated: {r['gpu_peak_allocated_mb']:.2f} MB",
                f"  Peak Cached: {r['gpu_peak_cached_mb']:.2f} MB",
            ])
            
        return "\n".join(summary)


def profile_memory_usage(func):
    """Decorator to profile memory usage of a function."""
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        with profiler.profile():
            result = func(*args, **kwargs)
        print(profiler.get_summary())
        return result
    return wrapper
