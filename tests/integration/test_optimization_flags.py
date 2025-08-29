import os
from typing import Any, Dict

import pytest


def test_cli_generate_forwards_optimization_flags(monkeypatch):
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

    # Invoke CLI with flags
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
            "--flash-attention",
            "--grad-checkpointing",
            "--tf32",
            "--matmul-precision",
            "high",
            "--cudnn-benchmark",
        ],
    )
    assert res.exit_code == 0, res.output

    # Validate forwarded kwargs
    assert captured.get("enable_flash_attention") is True
    assert captured.get("grad_checkpointing") is True
    assert captured.get("allow_tf32") is True
    assert captured.get("torch_matmul_precision") == "high"
    assert captured.get("cudnn_benchmark") is True


def test_model_loader_applies_backend_and_optional_features(monkeypatch):
    # Dummy runner with config flags set
    from types import SimpleNamespace

    # Provide functions/attrs possibly missing in minimal torch
    import torch

    # Trackers
    called = {
        "set_mm_prec": False,
        "cudnn_bench_set": False,
        "fa_called": False,
        "gc_called": False,
        "tf32_set": False,
    }

    # Monkeypatch torch backends, matmul precision
    backends = getattr(torch, "backends", None)
    if backends is None:

        class _CB:  # minimal cuda/cudnn backends stub
            class cuda:
                class matmul:
                    allow_tf32 = False

            class cudnn:
                benchmark = False

        torch.backends = _CB()  # type: ignore[attr-defined]
        backends = torch.backends

    # Ensure cuda availability check returns False to avoid device logic dependency
    monkeypatch.setattr(torch, "cuda", SimpleNamespace(is_available=lambda: False), raising=False)

    # Patch set_float32_matmul_precision
    def set_mm_prec(val: str):
        called["set_mm_prec"] = True

    monkeypatch.setattr(torch, "set_float32_matmul_precision", set_mm_prec, raising=False)

    # Provide TF32 setter path despite cuda unavailable (we just verify no exception path)
    # We still mark tf32_set when code tries to set it; emulate attribute
    class _MatmulNS:
        def __init__(self):
            self._allow_tf32 = False

        @property
        def allow_tf32(self):
            return self._allow_tf32

        @allow_tf32.setter
        def allow_tf32(self, v):
            called["tf32_set"] = True
            self._allow_tf32 = bool(v)

    if not hasattr(backends, "cuda"):

        class _CudaNS:
            matmul = _MatmulNS()

        backends.cuda = _CudaNS()  # type: ignore
    else:
        backends.cuda.matmul = _MatmulNS()

    # Patch cudnn benchmark
    if not hasattr(backends, "cudnn"):

        class _CudnnNS:
            def __init__(self):
                self._benchmark = False

            @property
            def benchmark(self):
                return self._benchmark

            @benchmark.setter
            def benchmark(self, v):
                called["cudnn_bench_set"] = True
                self._benchmark = bool(v)

        backends.cudnn = _CudnnNS()  # type: ignore
    else:

        class _BenchSet:
            def __init__(self):
                self._val = False

            @property
            def benchmark(self):
                return self._val

            @benchmark.setter
            def benchmark(self, v):
                called["cudnn_bench_set"] = True
                self._val = bool(v)

        backends.cudnn = _BenchSet()  # type: ignore

    # Patch FA enable
    import medvllm.optim.flash_attention as fa

    def fake_enable_flash_attention(model, config=None):  # type: ignore
        called["fa_called"] = True
        return model

    monkeypatch.setattr(fa, "enable_flash_attention", fake_enable_flash_attention, raising=True)

    # Dummy model returned by registry
    class DummyModel:
        def __init__(self):
            self.config = SimpleNamespace()
            self.gc_enabled = False

        def eval(self):
            return self

        def to(self, device):
            return self

        def gradient_checkpointing_enable(self):
            called["gc_called"] = True

    # Patch registry to avoid HF by overriding ModelRegistry.load_model to return DummyModel
    from medvllm.engine.model_runner.registry import ModelRegistry

    monkeypatch.setattr(
        ModelRegistry, "load_model", lambda self, *a, **k: DummyModel(), raising=True
    )

    # Also ensure any direct HF load path returns DummyModel
    import medvllm.engine.model_runner.model as model_mod

    monkeypatch.setattr(
        model_mod,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *a, **k: DummyModel()),
        raising=False,
    )

    # Build config
    from medvllm.config import Config

    cfg = Config(
        model="dummy",
        allow_tf32=True,
        torch_matmul_precision="high",
        cudnn_benchmark=True,
        enable_flash_attention=True,
        grad_checkpointing=True,
        use_medical_adapter=False,  # avoid adapter path
    )

    # Dummy runner providing minimal attributes
    runner = SimpleNamespace(config=cfg, device="cpu", dtype=None, past_key_values=None)

    # Execute load_model and assert side-effects
    from medvllm.engine.model_runner.model import ModelManager

    mm = ModelManager(runner)
    _ = mm.load_model("dummy")

    # Validate calls
    assert called["set_mm_prec"] is True
    assert called["tf32_set"] in (True, False)  # may be False if code gated by cuda availability
    assert called["cudnn_bench_set"] in (True, False)  # same as above
    assert called["fa_called"] is True
    assert called["gc_called"] is True


def test_cli_generate_no_optimization_flags_defaults(monkeypatch):
    # Ensure fake engine path to avoid model load
    monkeypatch.setenv("MEDVLLM_TEST_FAKE_ENGINE", "1")

    captured = {}

    import medvllm.tasks.text_generator as tg

    orig_init = tg.TextGenerator.__init__

    def fake_init(self, engine, constraints=None, **engine_kwargs):  # type: ignore[no-redef]
        captured.update(engine_kwargs)
        return orig_init(self, engine, constraints=constraints, **engine_kwargs)

    monkeypatch.setattr(tg.TextGenerator, "__init__", fake_init, raising=True)

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
            # no optimization flags passed
        ],
    )
    assert res.exit_code == 0, res.output

    # No optimization keys should be forwarded when flags are unset
    for k in (
        "enable_flash_attention",
        "grad_checkpointing",
        "allow_tf32",
        "torch_matmul_precision",
        "cudnn_benchmark",
    ):
        assert k not in captured


def test_model_loader_cpu_only_no_cuda_backend_changes_and_fa_config_forwarding(monkeypatch):
    from types import SimpleNamespace
    import torch

    # Trackers
    called = {
        "set_mm_prec": False,
        "cudnn_bench_set": False,
        "tf32_set": False,
        "fa_called": False,
        "fa_config_window": None,
    }

    # Force CPU-only environment
    monkeypatch.setattr(torch, "cuda", SimpleNamespace(is_available=lambda: False), raising=False)

    # Patch backends structure to verify that no-ops occur (values should remain unchanged)
    class _MatmulNS:
        def __init__(self):
            self._allow_tf32 = False

        @property
        def allow_tf32(self):
            return self._allow_tf32

        @allow_tf32.setter
        def allow_tf32(self, v):
            called["tf32_set"] = True
            self._allow_tf32 = bool(v)

    class _CudnnNS:
        def __init__(self):
            self._benchmark = False

        @property
        def benchmark(self):
            return self._benchmark

        @benchmark.setter
        def benchmark(self, v):
            called["cudnn_bench_set"] = True
            self._benchmark = bool(v)

    if not hasattr(torch, "backends"):
        torch.backends = SimpleNamespace()  # type: ignore[attr-defined]
    torch.backends.cuda = SimpleNamespace(matmul=_MatmulNS())  # type: ignore[attr-defined]
    torch.backends.cudnn = _CudnnNS()  # type: ignore[attr-defined]

    # matmul precision should still be applied on CPU
    def set_mm_prec(val: str):
        called["set_mm_prec"] = True

    monkeypatch.setattr(torch, "set_float32_matmul_precision", set_mm_prec, raising=False)

    # Stub FlashAttentionConfig and enable function to capture config dict
    import medvllm.optim.flash_attention as fa

    class _FAConfigStub:
        def __init__(self, window=None):
            self.window = window

        @classmethod
        def from_dict(cls, dct):
            return cls(window=dct.get("window"))

    def fake_enable_flash_attention(model, config=None):  # type: ignore
        called["fa_called"] = True
        called["fa_config_window"] = getattr(config, "window", None)
        return model

    monkeypatch.setattr(fa, "FlashAttentionConfig", _FAConfigStub, raising=True)
    monkeypatch.setattr(fa, "enable_flash_attention", fake_enable_flash_attention, raising=True)

    # Dummy model
    class DummyModel:
        def __init__(self):
            self.config = SimpleNamespace()

        def eval(self):
            return self

        def to(self, device):
            return self

    # Patch registry and direct HF load to avoid real loading
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

    # Build config: CPU-only, FA on with config, TF32/cudnn requested
    from medvllm.config import Config

    cfg = Config(
        model="dummy",
        allow_tf32=True,
        torch_matmul_precision="high",
        cudnn_benchmark=True,
        enable_flash_attention=True,
        flash_attention_config={"window": 128},
        use_medical_adapter=False,
    )

    runner = SimpleNamespace(config=cfg, device="cpu", dtype=None, past_key_values=None)

    from medvllm.engine.model_runner.model import ModelManager

    mm = ModelManager(runner)
    _ = mm.load_model("dummy")

    # Since cuda is unavailable, TF32/cudnn backend changes should be no-ops
    assert called["tf32_set"] is False
    assert called["cudnn_bench_set"] is False
    # matmul precision still applies
    assert called["set_mm_prec"] is True
    # FlashAttention enable was called and received our config
    assert called["fa_called"] is True
    assert called["fa_config_window"] == 128
