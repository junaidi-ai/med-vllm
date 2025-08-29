import os
import types
from typing import Any, Dict

import torch
import pytest
import torch.nn as nn

try:
    import torch.nn.functional as F  # type: ignore

    _relu = getattr(F, "relu", None)
    if not callable(_relu):
        raise ImportError
except Exception:

    def _relu(x):  # type: ignore
        return x


from click.testing import CliRunner

from medvllm.cli import cli as root_cli
from medvllm.optim.quantization import bnb_load_quantized, quantize_model, QuantizationConfig


def test_cli_generate_accepts_quantization_flags(monkeypatch: Any) -> None:
    # Use FakeEngine path to avoid real model loading
    monkeypatch.setenv("MEDVLLM_TEST_FAKE_ENGINE", "1")

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "generate",
            "--text",
            "Test quantization flags",
            "--model",
            "dummy-model",
            "--quantization-bits",
            "4",
            "--quantization-method",
            "bnb-nf4",
        ],
    )  # type: ignore[arg-type]
    assert res.exit_code == 0, res.output
    assert "FAKE ->" in res.output  # ensures FakeEngine path executed


def test_cli_generate_forwards_quantization_flags(monkeypatch: Any) -> None:
    # Ensure fake engine path to avoid model load
    monkeypatch.setenv("MEDVLLM_TEST_FAKE_ENGINE", "1")

    captured: Dict[str, Any] = {}

    # Patch TextGenerator.__init__ to capture engine kwargs
    import medvllm.tasks.text_generator as tg

    orig_init = tg.TextGenerator.__init__

    def fake_init(self, engine, constraints=None, **engine_kwargs):  # type: ignore[no-redef]
        captured.update(engine_kwargs)
        return orig_init(self, engine, constraints=constraints, **engine_kwargs)

    monkeypatch.setattr(tg.TextGenerator, "__init__", fake_init, raising=True)

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
            "--quantization-bits",
            "4",
            "--quantization-method",
            "bnb-nf4",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured.get("quantization_bits") == 4
    assert captured.get("quantization_method") == "bnb-nf4"


def test_bnb_load_quantized_builds_expected_kwargs(monkeypatch: Any) -> None:
    captured: Dict[str, Any] = {}

    class DummyModel:
        pass

    def fake_from_pretrained(name: str, **kwargs: Any) -> DummyModel:  # type: ignore[override]
        captured.clear()
        captured.update(kwargs)
        return DummyModel()

    # Monkeypatch transformers path
    import medvllm.optim.quantization as q

    monkeypatch.setattr(
        q.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained, raising=True
    )

    # 4-bit NF4
    _ = bnb_load_quantized("foo/bar", bits=4, method="bnb-nf4", device_map="auto")
    assert captured.get("load_in_4bit") is True
    assert captured.get("bnb_4bit_quant_type") == "nf4"
    assert captured.get("device_map") == "auto"

    # 8-bit
    _ = bnb_load_quantized("foo/bar", bits=8, method="bnb-8bit", device_map=None)
    assert captured.get("load_in_8bit") is True
    assert "device_map" not in captured  # None should not set device_map


def test_dynamic_quantize_model_forward_runs() -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(8)
            self.fc = nn.Linear(8, 8)
            self.ln2 = nn.LayerNorm(8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.ln2(_relu(self.fc(self.ln1(x))))

    # Some minimal torch environments may not expose manual_seed
    seed_fn = getattr(torch, "manual_seed", None)
    if callable(seed_fn):
        try:
            seed_fn(0)
        except Exception:
            pass
    m = Tiny().eval()

    qcfg = QuantizationConfig(dtype=torch.qint8, inplace=False)
    mq = quantize_model(m, qcfg)

    x = torch.randn(2, 8)
    inf_mode = getattr(torch, "inference_mode", None)

    def _run(m):
        # Try callable, then .forward, else skip
        if callable(m):
            return m(x)
        fwd = getattr(m, "forward", None)
        if callable(fwd):
            return fwd(x)
        pytest.skip("Minimal torch environment: module not callable and no .forward available")

    if callable(inf_mode):
        with inf_mode():
            y = _run(mq)
    else:
        # Fallback for minimal torch
        no_grad = getattr(torch, "no_grad", None)
        if callable(no_grad):
            with no_grad():
                y = _run(mq)
        else:
            y = _run(mq)
    assert y.shape == (2, 8)


def test_model_loader_dynamic_quantization_applied_when_requested(monkeypatch: Any) -> None:
    # Spy on quantize_model to ensure it is invoked
    calls: Dict[str, Any] = {"called": False, "dtype": None}

    import medvllm.optim.quantization as q

    real_quantize_model = q.quantize_model

    def spy_quantize_model(model, cfg, *a, **k):  # type: ignore[no-redef]
        calls["called"] = True
        calls["dtype"] = getattr(cfg, "dtype", None)
        return model

    monkeypatch.setattr(q, "quantize_model", spy_quantize_model, raising=True)

    # Dummy model and loader patches
    from types import SimpleNamespace

    class DummyModel:
        def __init__(self):
            self.config = SimpleNamespace()

        def eval(self):
            return self

        def to(self, device):
            return self

    from medvllm.engine.model_runner.registry import ModelRegistry

    monkeypatch.setattr(
        ModelRegistry, "load_model", lambda self, *a, **k: DummyModel(), raising=True
    )

    import medvllm.engine.model_runner.model as model_mod

    monkeypatch.setattr(
        model_mod,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *a, **k: DummyModel()),
        raising=False,
    )

    from medvllm.config import Config

    cfg = Config(
        model="dummy",
        quantization_bits=8,
        quantization_method="dynamic",
        use_medical_adapter=False,
    )

    runner = SimpleNamespace(config=cfg, device="cpu", dtype=None, past_key_values=None)

    from medvllm.engine.model_runner.model import ModelManager

    mm = ModelManager(runner)
    _ = mm.load_model("dummy")

    assert calls["called"] is True
    # Expect qint8 for dynamic quantization by default
    assert calls["dtype"] in (torch.qint8, getattr(torch, "qint8", None))


def test_model_loader_bnb_kwargs_forwarded(monkeypatch: Any) -> None:
    # Capture kwargs passed to HF from_pretrained via bnb path
    captured: Dict[str, Any] = {}

    from types import SimpleNamespace

    class DummyModel:
        def __init__(self):
            self.config = SimpleNamespace()

        def eval(self):
            return self

        def to(self, device):
            return self

    def fake_from_pretrained(*a, **k):
        captured.clear()
        captured.update(k)
        return DummyModel()

    # Patch transformers class method (ModelManager imports inside function)
    import transformers as _t

    monkeypatch.setattr(
        _t.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained, raising=False
    )

    # Ensure registry path doesn't shadow the bnb early path
    from medvllm.engine.model_runner.registry import ModelRegistry

    monkeypatch.setattr(ModelRegistry, "load_model", lambda self, *a, **k: None, raising=True)

    from medvllm.config import Config

    cfg = Config(
        model="dummy",
        quantization_bits=4,
        quantization_method="bnb-nf4",
        use_medical_adapter=False,
    )

    runner = SimpleNamespace(config=cfg, device="cuda", dtype=None, past_key_values=None)

    from medvllm.engine.model_runner.model import ModelManager

    mm = ModelManager(runner)
    _ = mm.load_model("dummy")

    # Validate bitsandbytes kwargs
    assert captured.get("load_in_4bit") is True
    assert captured.get("bnb_4bit_quant_type") == "nf4"
    assert captured.get("device_map") == "auto"
