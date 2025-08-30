import os
from types import SimpleNamespace
from typing import Any, Dict

import pytest


def test_cli_generate_forwards_memory_pooling_flags(monkeypatch):
    # Ensure fake engine to avoid heavy deps
    monkeypatch.setenv("MEDVLLM_TEST_FAKE_ENGINE", "1")

    captured: Dict[str, Any] = {}

    # Patch TextGenerator.__init__ to capture engine kwargs
    import medvllm.tasks.text_generator as tg

    orig_init = tg.TextGenerator.__init__

    def fake_init(self, engine, constraints=None, **engine_kwargs):  # type: ignore[no-redef]
        captured.update(engine_kwargs)
        return orig_init(self, engine, constraints=constraints, **engine_kwargs)

    monkeypatch.setattr(tg.TextGenerator, "__init__", fake_init, raising=True)

    # Invoke CLI with pooling flags
    from click.testing import CliRunner
    from medvllm.cli import cli as root_cli

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "generate",
            "--text",
            "Hello",
            "--model",
            "dummy",
            "--memory-pooling",
            "--pool-max-bytes",
            "1048576",
            "--pool-device",
            "cuda",
        ],
    )
    assert res.exit_code == 0, res.output

    # Validate forwarded kwargs
    assert captured.get("enable_memory_pooling") is True
    assert captured.get("pool_max_bytes") == 1048576
    assert captured.get("pool_device") == "cuda"


def test_model_manager_merges_pooling_into_adapter_config(monkeypatch):
    # Dummy runner with config flags set
    from medvllm.config import Config

    cfg = Config(
        model="dummy",
        enable_memory_pooling=True,
        pool_max_bytes=2_000_000,
        pool_device="cpu",
        use_medical_adapter=True,
    )

    # Stub model and registry to avoid HF load
    class DummyModel:
        def __init__(self):
            self.config = SimpleNamespace()

        def eval(self):
            return self

        def to(self, device):  # noqa: ARG002
            return self

    from medvllm.engine.model_runner.registry import ModelRegistry

    monkeypatch.setattr(
        ModelRegistry, "load_model", lambda self, *a, **k: DummyModel(), raising=True
    )

    # Capture adapter_config passed to AdapterManager.create_adapter
    captured: Dict[str, Any] = {}

    import medvllm.models.adapter_manager as am

    def fake_create_adapter(*, model, model_name_or_path, adapter_type, adapter_config, hf_config):  # type: ignore[no-redef]
        # Record and return a simple stub
        captured.update(dict(adapter_config=adapter_config, adapter_type=adapter_type))
        return SimpleNamespace(apply=lambda *a, **k: None)

    monkeypatch.setattr(am.AdapterManager, "create_adapter", staticmethod(fake_create_adapter))

    # Build a minimal runner namespace
    runner = SimpleNamespace(config=cfg, device="cpu", dtype=None, past_key_values=None)

    from medvllm.engine.model_runner.model import ModelManager

    mm = ModelManager(runner)
    _ = mm.load_model("dummy")

    # Assert pooling flags are present in adapter_config
    adapter_cfg = captured.get("adapter_config", {})
    assert adapter_cfg.get("enable_memory_pooling") is True
    assert adapter_cfg.get("pool_max_bytes") == 2_000_000
    assert adapter_cfg.get("pool_device") == "cpu"
