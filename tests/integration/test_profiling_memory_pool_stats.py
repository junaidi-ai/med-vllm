from __future__ import annotations

import types
from typing import Any, Dict

import torch
import pytest

from medvllm.engine.model_runner.base import ModelRunner


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None, **kwargs: Any
    ):
        # Return a minimal logits-like object compatible with our mocks without relying on shapes
        logits = torch.zeros(1) if hasattr(torch, "zeros") else None
        return types.SimpleNamespace(logits=logits, past_key_values=None)

    # Some torch mocks don't wire __call__ -> forward; provide explicitly
    def __call__(self, *args, **kwargs):  # type: ignore[override]
        return self.forward(*args, **kwargs)


class _DummyRunner(ModelRunner):
    def _initialize_model(self) -> None:  # minimal model on selected device/dtype
        self.model_manager.model = _DummyModel().to(self.device)

    def _initialize_components(self) -> None:
        # Nothing extra needed for this test
        pass


@pytest.mark.parametrize("enable_pool", [True, False])
def test_last_profile_contains_memory_pool_block(enable_pool: bool):
    # Minimal config object with needed attributes
    Cfg = type(
        "Cfg",
        (),
        dict(
            enable_profiling=True,
            profiler_device="cpu",  # keep lightweight in CI
            emit_trace=False,
            trace_dir=None,
            enable_memory_pooling=enable_pool,
        ),
    )
    cfg = Cfg()

    runner = _DummyRunner(config=cfg, world_size=1, rank=0, device="cpu")

    # Torch mock compatibility: ensure dtypes/functions exist
    if not hasattr(torch, "long") and hasattr(torch, "int64"):
        torch.long = torch.int64  # type: ignore[attr-defined]
    if not hasattr(torch, "empty") and hasattr(torch, "zeros"):
        torch.empty = lambda *args, **kwargs: torch.zeros(*args, **kwargs)  # type: ignore[assignment]

    # Prepare tiny inputs (avoid tensor methods not supported by mocks)
    input_ids = torch.zeros((1, 2), dtype=torch.long, device=runner.device)
    pos = None

    # Invoke the model path that records last_profile
    logits, pkv = runner.model_manager.run_model(input_ids, pos, is_prefill=True)

    # Assert last_profile exists and contains memory_pool section
    assert hasattr(runner, "last_profile"), "runner.last_profile was not set by run_model()"
    prof: Dict[str, Any] = getattr(runner, "last_profile")
    assert isinstance(prof, dict)
    assert "memory_pool" in prof, "memory_pool section missing in profiling results"
    mp = prof["memory_pool"]
    assert isinstance(mp, dict)
    # Basic keys expected from MemoryManager.pool_stats
    for key in (
        "acquire_requests",
        "acquire_hits",
        "acquire_misses",
        "released",
        "reused",
        "evicted",
        "bytes",
    ):
        assert key in mp, f"missing memory pool stat: {key}"

    # When pooling disabled, stats should still be present (all zeros)
    if not enable_pool:
        assert all(int(mp[k]) == 0 for k in mp if isinstance(mp[k], int))
