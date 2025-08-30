"""Reusable runtime profiler utility.

Prefers torch.profiler (CPU/GPU time and memory) and falls back to the
local MemoryProfiler used by benchmarks (tests.medical.memory_profiler).

API
---
- get_profiler(device: str = "auto", emit_trace: bool = False, trace_dir: Optional[str] = None) -> Prof
  Prof has:
    - profile() -> context manager
    - results: Dict[str, Any] available after exiting context

Results shape (best-effort):
{
  "device": "cpu"|"cuda",
  "trace_path": Optional[str],
  "cpu_time_total_ms": float,
  "gpu_time_total_ms": float,
  "cpu_max_rss_mb": float,
  "cuda_max_mem_mb": float,
}

This module has no external deps and will gracefully degrade.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class _BaseProfiler:
    device: str = "auto"
    emit_trace: bool = False
    trace_dir: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)

    def profile(self):  # pragma: no cover - simple wrapper
        return _NoOpCtx(self)


class _NoOpCtx:
    def __init__(self, prof: _BaseProfiler) -> None:
        self.prof = prof
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self.prof.results = {
            "device": self.prof.device,
            "cpu_time_total_ms": dt_ms,
            "gpu_time_total_ms": 0.0,
            "cpu_max_rss_mb": None,
            "cuda_max_mem_mb": None,
            "trace_path": None,
        }
        return False


class _TorchProfiler(_BaseProfiler):
    def profile(self):  # pragma: no cover - depends on torch install
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile
        except Exception:
            return _NoOpCtx(self)

        activities = [ProfilerActivity.CPU]
        if self.device == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        output_file = None
        if self.emit_trace:
            outdir = self.trace_dir or os.path.join(os.getcwd(), "profiles")
            os.makedirs(outdir, exist_ok=True)
            output_file = os.path.join(outdir, f"trace_{int(time.time())}.json")

        # Simple schedule for single-step measurement
        prof_cm = profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
            with_modules=False,
            on_trace_ready=(
                (lambda p: p.export_chrome_trace(output_file)) if output_file else None
            ),
        )

        base = self

        class _Ctx:
            def __enter__(self):
                self._t0 = time.perf_counter()
                prof_cm.__enter__()
                return self

            def __exit__(self, et, ev, tb):
                prof_cm.__exit__(et, ev, tb)
                dt_ms = (time.perf_counter() - self._t0) * 1000.0
                cpu_time = float(dt_ms)
                gpu_time = 0.0
                cpu_mem = None
                cuda_mem = None
                try:
                    # Aggregate best-effort stats
                    key_averages = prof_cm.key_averages()
                    if key_averages is not None:
                        # Sum CUDA time if present
                        cuda_total_us = sum(
                            getattr(ev, "cuda_time_total", 0.0) for ev in key_averages
                        )
                        gpu_time = float(cuda_total_us) / 1000.0  # us -> ms
                    # Memory
                    if base.device == "cuda" and torch.cuda.is_available():
                        cuda_mem = float(torch.cuda.max_memory_allocated()) / (1024**2)
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
                # Try RSS via psutil if present (best-effort, optional dep)
                try:
                    import psutil  # type: ignore

                    process = psutil.Process(os.getpid())
                    rss = process.memory_info().rss
                    cpu_mem = float(rss) / (1024**2)
                except Exception:
                    pass

                base.results = {
                    "device": base.device,
                    "cpu_time_total_ms": cpu_time,
                    "gpu_time_total_ms": gpu_time,
                    "cpu_max_rss_mb": cpu_mem,
                    "cuda_max_mem_mb": cuda_mem,
                    "trace_path": output_file,
                }
                return False

        return _Ctx()


class _BenchMemoryProfiler(_BaseProfiler):
    """Fallback to local tests.medical.memory_profiler if torch.profiler not used."""

    def profile(self):  # pragma: no cover
        try:
            from tests.medical.memory_profiler import MemoryProfiler  # type: ignore

            mem_prof = MemoryProfiler(device=self.device)
            ctx = mem_prof.profile()
            base = self

            class _Ctx:
                def __enter__(self):
                    return ctx.__enter__()

                def __exit__(self, et, ev, tb):
                    ctx.__exit__(et, ev, tb)
                    base.results = getattr(mem_prof, "results", {})
                    if base.results is None:
                        base.results = {}
                    base.results.setdefault("device", base.device)
                    base.results.setdefault("trace_path", None)
                    return False

            return _Ctx()
        except Exception:
            return _NoOpCtx(self)


def get_profiler(device: str = "auto", emit_trace: bool = False, trace_dir: Optional[str] = None):
    """Return a profiler instance.

    Tries torch.profiler first; if that fails, tries local MemoryProfiler; otherwise no-op.
    """
    dev = (device or "auto").lower()
    if dev == "auto":
        try:
            import torch  # type: ignore

            dev = "cuda" if (torch.cuda.is_available()) else "cpu"
        except Exception:
            dev = "cpu"

    # Use torch profiler by default when available
    try:
        import torch  # noqa: F401
        from torch import profiler as _p  # noqa: F401

        return _TorchProfiler(device=dev, emit_trace=emit_trace, trace_dir=trace_dir)
    except Exception:
        # Fallback to benchmark memory profiler
        return _BenchMemoryProfiler(device=dev, emit_trace=emit_trace, trace_dir=trace_dir)
