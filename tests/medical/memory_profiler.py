"""Simple memory profiler utility used by benchmarks.

Provides a context manager that records CPU and (optionally) GPU memory
at the beginning and end of a code section and exposes a `results` dict
with deltas and raw values in megabytes.

Expected API (used by benchmarks/benchmark_medical.py):
- MemoryProfiler(device: str)
- with mem_profiler.profile(): ...
- mem_profiler.results -> Dict[str, float]

Keys present when CUDA available and device == 'cuda':
- gpu_allocated_start_mb, gpu_allocated_end_mb, gpu_allocated_delta_mb
- gpu_reserved_start_mb,  gpu_reserved_end_mb,  gpu_reserved_delta_mb
- gpu_cached_start_mb,    gpu_cached_end_mb    (alias of reserved metrics)

CPU keys (always present):
- cpu_rss_start_mb, cpu_rss_end_mb, cpu_rss_delta_mb
- cpu_vms_start_mb, cpu_vms_end_mb, cpu_vms_delta_mb
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import psutil
import torch


_MB = 1024 * 1024


class MemoryProfiler:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.results: Dict[str, float] = {}
        self._start: Dict[str, float] = {}
        self._end: Dict[str, float] = {}

    def _snapshot_cpu(self, dest: Dict[str, float]) -> None:
        proc = psutil.Process()
        mem = proc.memory_info()
        dest["cpu_rss_mb"] = mem.rss / _MB
        dest["cpu_vms_mb"] = mem.vms / _MB

    def _snapshot_gpu(self, dest: Dict[str, float]) -> None:
        if self.device != "cuda" or not torch.cuda.is_available():
            return
        # Ensure queued work is completed before sampling
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        dest["gpu_allocated_mb"] = torch.cuda.memory_allocated() / _MB
        dest["gpu_reserved_mb"] = torch.cuda.memory_reserved() / _MB
        # For compatibility with benchmark printing that refers to cached_* keys
        dest["gpu_cached_mb"] = dest["gpu_reserved_mb"]

    def _snapshot_start(self) -> None:
        self._start = {}
        self._snapshot_cpu(self._start)
        self._snapshot_gpu(self._start)

    def _snapshot_end(self) -> None:
        self._end = {}
        self._snapshot_cpu(self._end)
        self._snapshot_gpu(self._end)

    def _compute_results(self) -> None:
        results: Dict[str, float] = {}
        # CPU deltas
        results["cpu_rss_start_mb"] = self._start.get("cpu_rss_mb", 0.0)
        results["cpu_rss_end_mb"] = self._end.get("cpu_rss_mb", 0.0)
        results["cpu_rss_delta_mb"] = results["cpu_rss_end_mb"] - results["cpu_rss_start_mb"]

        results["cpu_vms_start_mb"] = self._start.get("cpu_vms_mb", 0.0)
        results["cpu_vms_end_mb"] = self._end.get("cpu_vms_mb", 0.0)
        results["cpu_vms_delta_mb"] = results["cpu_vms_end_mb"] - results["cpu_vms_start_mb"]

        # GPU deltas (if available)
        if "gpu_allocated_mb" in self._start and "gpu_allocated_mb" in self._end:
            results["gpu_allocated_start_mb"] = self._start["gpu_allocated_mb"]
            results["gpu_allocated_end_mb"] = self._end["gpu_allocated_mb"]
            results["gpu_allocated_delta_mb"] = (
                results["gpu_allocated_end_mb"] - results["gpu_allocated_start_mb"]
            )

        if "gpu_reserved_mb" in self._start and "gpu_reserved_mb" in self._end:
            results["gpu_reserved_start_mb"] = self._start["gpu_reserved_mb"]
            results["gpu_reserved_end_mb"] = self._end["gpu_reserved_mb"]
            results["gpu_reserved_delta_mb"] = (
                results["gpu_reserved_end_mb"] - results["gpu_reserved_start_mb"]
            )
            # Alias to cached_* for compatibility with benchmark printout
            results["gpu_cached_start_mb"] = results["gpu_reserved_start_mb"]
            results["gpu_cached_end_mb"] = results["gpu_reserved_end_mb"]

        self.results = results

    @contextmanager
    def profile(self):
        """Context manager to profile memory around a code block."""
        self._snapshot_start()
        try:
            yield
        finally:
            # Try to synchronize again before final sampling for accuracy
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            self._snapshot_end()
            self._compute_results()
