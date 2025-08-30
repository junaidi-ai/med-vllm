import os
from typing import Any, Dict

import pytest

from medvllm.configs.profiles import (
    resolve_profile_engine_kwargs,
    load_profile,
    profile_engine_kwargs,
)
from medvllm.cli.inference_commands import _build_engine_kwargs


@pytest.mark.parametrize(
    "name, expect_bits, expect_method",
    [
        ("cpu", 8, "dynamic"),
        ("gpu_8bit", 8, "bnb-8bit"),
        ("gpu_4bit", 4, "bnb-nf4"),
    ],
)
def test_resolve_profile_engine_kwargs_by_name(
    name: str, expect_bits: int, expect_method: str
) -> None:
    kw = resolve_profile_engine_kwargs(name)
    # 'device' is present in JSON but must be filtered out
    assert "device" not in kw
    assert kw.get("quantization_bits") == expect_bits
    assert kw.get("quantization_method") == expect_method


def test_resolve_profile_engine_kwargs_by_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("MEDVLLM_DEPLOYMENT_PROFILE", "gpu_8bit")
    kw = resolve_profile_engine_kwargs(None)
    assert kw.get("quantization_bits") == 8
    assert kw.get("quantization_method") == "bnb-8bit"


def test_resolve_profile_invalid_path_raises(tmp_path) -> None:
    bogus = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        _ = load_profile(str(bogus))


def test_profile_engine_kwargs_filters_unknown_keys() -> None:
    data = {
        "engine": {
            "device": "cuda",  # not a Config field
            "tensor_parallel_size": 2,
            "quantization_bits": 4,
        }
    }
    kw = profile_engine_kwargs(data)
    assert "device" not in kw
    assert kw["tensor_parallel_size"] == 2
    assert kw["quantization_bits"] == 4


def test_build_engine_kwargs_profile_then_override(monkeypatch: Any) -> None:
    # Start from cpu profile via name
    out = _build_engine_kwargs(deployment_profile="cpu")
    assert out.get("quantization_bits") == 8
    assert out.get("quantization_method") == "dynamic"

    # Override via explicit args must take precedence
    out2 = _build_engine_kwargs(
        deployment_profile="cpu", quantization_bits=4, quantization_method="bnb-nf4"
    )
    assert out2.get("quantization_bits") == 4
    assert out2.get("quantization_method") == "bnb-nf4"


def test_build_engine_kwargs_env_profile(monkeypatch: Any) -> None:
    monkeypatch.setenv("MEDVLLM_DEPLOYMENT_PROFILE", "gpu_4bit")
    out = _build_engine_kwargs(deployment_profile=None)
    assert out.get("quantization_bits") == 4
    assert out.get("quantization_method") == "bnb-nf4"


def test_build_engine_kwargs_invalid_profile_warns_and_continues(monkeypatch: Any) -> None:
    # Capture warnings by swapping warn function in module
    import medvllm.cli.inference_commands as ic

    messages: list[str] = []

    def _capture(msg: str) -> None:
        messages.append(str(msg))

    monkeypatch.setattr(ic, "warn", _capture, raising=True)

    out = _build_engine_kwargs(deployment_profile="/path/does/not/exist.json", quantization_bits=8)
    assert out.get("quantization_bits") == 8  # explicit arg still present
    # Should have emitted a warning about profile not found
    assert any("Deployment profile not found" in m for m in messages)


def test_build_engine_kwargs_conflicting_attention_flags_warn(monkeypatch: Any) -> None:
    import medvllm.cli.inference_commands as ic

    messages: list[str] = []

    def _capture(msg: str) -> None:
        messages.append(str(msg))

    monkeypatch.setattr(ic, "warn", _capture, raising=True)

    _ = _build_engine_kwargs(attention_impl="flash", enable_flash_attention=False)
    assert any("--attention-impl flash" in m for m in messages)


def test_build_engine_kwargs_memory_pooling_warnings(monkeypatch: Any) -> None:
    import medvllm.cli.inference_commands as ic

    messages: list[str] = []

    def _capture(msg: str) -> None:
        messages.append(str(msg))

    monkeypatch.setattr(ic, "warn", _capture, raising=True)

    _ = _build_engine_kwargs(enable_memory_pooling=False, pool_max_bytes=1024)
    assert any("Memory pooling options" in m for m in messages)


def test_profile_edge_cpu_int8_maps_expected_fields() -> None:
    kw = resolve_profile_engine_kwargs("edge_cpu_int8")
    # filtered fields only
    assert kw.get("tensor_parallel_size") == 1
    assert kw.get("quantization_bits") == 8
    assert kw.get("quantization_method") == "dynamic"
    # memory efficiency and pooling retained
    assert kw.get("memory_efficient") is True
    assert kw.get("enable_memory_pooling") is True
    assert kw.get("pool_device") == "auto"


def test_profile_onprem_gpu_bf16_flash_maps_expected_fields() -> None:
    kw = resolve_profile_engine_kwargs("onprem_gpu_bf16_flash")
    assert kw.get("tensor_parallel_size") == 1
    assert kw.get("enable_mixed_precision") is True
    assert kw.get("mixed_precision_dtype") == "bf16"
    assert kw.get("enable_flash_attention") is True
    assert kw.get("attention_impl") == "flash"
    assert kw.get("allow_tf32") is True
    assert kw.get("cudnn_benchmark") is True


def test_profile_cloud_gpu_tp4_fp16_maps_expected_fields() -> None:
    kw = resolve_profile_engine_kwargs("cloud_gpu_tp4_fp16")
    assert kw.get("tensor_parallel_size") == 4
    assert kw.get("enable_mixed_precision") is True
    assert kw.get("mixed_precision_dtype") == "fp16"
    assert kw.get("enable_memory_pooling") is True
    assert kw.get("pool_device") == "cuda"


def test_profile_schema_missing_engine_raises(tmp_path) -> None:
    p = tmp_path / "bad_profile.json"
    p.write_text(
        '{"profile": "bad", "category": "edge", "description": "no engine"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _ = load_profile(str(p))


def test_profile_schema_invalid_category_raises(tmp_path) -> None:
    p = tmp_path / "bad_cat.json"
    p.write_text(
        '{"profile": "bad", "category": "datacenter", "engine": {"tensor_parallel_size": 1}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _ = load_profile(str(p))
