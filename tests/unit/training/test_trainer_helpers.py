import os
from typing import Dict, Any

import pytest
import torch

from medvllm.training.trainer import MedicalModelTrainer, TrainerConfig, _EarlyStopSignal


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)

    def forward(self, **batch: Dict[str, Any]):
        # Return a fake loss for trainer pathways that might call it
        x = batch.get("x", torch.randn(1, 2))
        y = self.lin(x).sum()
        return {"loss": y}


@pytest.fixture
def trainer_cpu():
    model = DummyModel()
    cfg = TrainerConfig()
    t = MedicalModelTrainer(model, cfg)
    # Force CPU for predictable tests
    t.device = torch.device("cpu")
    return t


def test_single_process_helpers(trainer_cpu, monkeypatch):
    # Ensure single process env
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    # torch.distributed not initialized
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    assert trainer_cpu._is_distributed() is False
    assert trainer_cpu._should_use_ddp() is False
    assert trainer_cpu._should_use_deepspeed() is False
    assert trainer_cpu._is_distributed_initialized() is False
    # rank0 should default to local rank 0 when not initialized
    assert trainer_cpu._is_rank0() is True


def test_mocked_ddp_init_and_wrap(trainer_cpu, monkeypatch):
    # Simulate distributed environment via WORLD_SIZE
    monkeypatch.setenv("WORLD_SIZE", "2")

    import torch.distributed as dist

    # Track calls
    called = {"init": False, "ddp_wrap": False}

    def fake_init_process_group(*args, **kwargs):
        called["init"] = True

    # dist availability and init state
    monkeypatch.setattr(dist, "is_available", lambda: True)
    # Before init, report not initialized; after calling our fake init, we will flip via closure
    init_state = {"initialized": False}

    def fake_is_initialized():
        return init_state["initialized"]

    def wrapped_init(*args, **kwargs):
        fake_init_process_group(*args, **kwargs)
        init_state["initialized"] = True

    monkeypatch.setattr(dist, "is_initialized", fake_is_initialized)
    monkeypatch.setattr(dist, "init_process_group", wrapped_init)

    # Mock DDP wrapper to just set a flag and return the model
    class FakeDDP:
        def __init__(self, module, **kwargs):
            called["ddp_wrap"] = True
            self.module = module

        def __getattr__(self, item):
            return getattr(self.module, item)

    # Ensure torch.nn.parallel exists in the torch stub
    if not hasattr(torch.nn, "parallel"):

        class _ParallelNS:
            pass

        torch.nn.parallel = _ParallelNS()

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP, raising=False)

    # Now run the trainer hooks
    assert trainer_cpu._should_use_ddp() is True
    trainer_cpu._init_distributed()
    assert called["init"] is True
    # After init, _wrap_model_ddp should engage
    trainer_cpu._wrap_model_ddp()
    assert called["ddp_wrap"] is True
    assert trainer_cpu._wrapped_with_ddp is True

    # _is_rank0 should consult dist.get_rank when initialized
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)
    assert trainer_cpu._is_rank0() is True
    monkeypatch.setattr(dist, "get_rank", lambda: 1, raising=False)
    assert trainer_cpu._is_rank0() is False


def test_deepspeed_flag_fallback_no_install(trainer_cpu, monkeypatch):
    # Enable deepspeed in config but ensure import fails
    trainer_cpu.config.deepspeed = True

    # Simulate missing deepspeed by making import fail
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "deepspeed":
            raise ImportError("No module named 'deepspeed'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    # Also ensure DDP isn't forced to reduce side effects in this test
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    # Should not raise; should warn and return gracefully
    trainer_cpu._init_deepspeed()


def test_early_stop_signal_is_bare_exception():
    sig = _EarlyStopSignal()
    # These helpers should not exist on the exception class
    for name in [
        "_is_distributed",
        "_local_rank",
        "_is_rank0",
        "_should_use_ddp",
        "_should_use_deepspeed",
        "_init_distributed",
        "_wrap_model_ddp",
        "_init_deepspeed",
        "_is_distributed_initialized",
        "_wrap_model_fsdp",
        "_should_use_fsdp",
        "_load_checkpoint",
        "_export_artifacts",
        "_next_version",
    ]:
        assert not hasattr(sig, name), f"_EarlyStopSignal unexpectedly has attribute {name}"
