import json

import torch

from tests.medical.memory_profiler import MemoryProfiler


def test_memory_profiler_cpu_keys_present():
    mp = MemoryProfiler(device="cpu")
    with mp.profile():
        x = sum(i for i in range(10000))
        assert x >= 0
    res = mp.results
    # Required CPU keys
    for k in [
        "cpu_rss_start_mb",
        "cpu_rss_end_mb",
        "cpu_rss_delta_mb",
        "cpu_vms_start_mb",
        "cpu_vms_end_mb",
        "cpu_vms_delta_mb",
    ]:
        assert k in res, f"Missing key: {k} in results: {json.dumps(res, indent=2)}"
    # Values are numbers
    assert isinstance(res["cpu_rss_start_mb"], (int, float))
    assert isinstance(res["cpu_rss_end_mb"], (int, float))
    assert isinstance(res["cpu_rss_delta_mb"], (int, float))


def test_memory_profiler_gpu_keys_optional():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp = MemoryProfiler(device=device)
    with mp.profile():
        # Allocate a small tensor on the chosen device
        t = torch.randn(8, 8, device=device)
        del t
    res = mp.results

    if device == "cuda" and torch.cuda.is_available():
        # GPU keys should be present when profiling on CUDA
        for k in [
            "gpu_allocated_start_mb",
            "gpu_allocated_end_mb",
            "gpu_allocated_delta_mb",
            "gpu_reserved_start_mb",
            "gpu_reserved_end_mb",
            "gpu_reserved_delta_mb",
            "gpu_cached_start_mb",
            "gpu_cached_end_mb",
        ]:
            assert k in res, f"Missing GPU key: {k} in results: {json.dumps(res, indent=2)}"
        assert isinstance(res["gpu_allocated_end_mb"], (int, float))
    else:
        # On CPU-only, GPU keys may be absent
        assert "gpu_allocated_end_mb" not in res or isinstance(
            res.get("gpu_allocated_end_mb"), (int, float)
        )
