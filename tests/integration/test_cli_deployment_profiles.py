import json
from typing import Any, Dict, List

from click.testing import CliRunner

from medvllm.cli import cli as root_cli


class _Capture:
    last_init_args: Dict[str, Any] | None = None


class DummyLLM:
    def __init__(self, model: str, **kwargs: Any) -> None:
        # capture all engine kwargs
        _Capture.last_init_args = {"model": model, **kwargs}

    def generate(self, prompts: List[str], sampling_params: Any, use_tqdm: bool = False):  # type: ignore[no-untyped-def]
        text = f"OK:{prompts[0][:32]}"
        return [{"text": text, "token_ids": [1, 2, 3]}]


def test_cli_generate_profile_and_overrides_reach_engine(monkeypatch: Any) -> None:
    # Patch LLM used inside TextGenerator to capture kwargs
    import medvllm.tasks.text_generator as tg

    monkeypatch.setattr(tg, "LLM", DummyLLM, raising=True)

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "generate",
            "--text",
            "Hello",
            "--model",
            "dummy-model",
            # apply profile defaults first
            "--deployment-profile",
            "cpu",
            # then override via CLI
            "--quantization-bits",
            "4",
            "--quantization-method",
            "bnb-nf4",
            "--json-meta",
        ],
    )  # type: ignore[arg-type]
    assert res.exit_code == 0, res.output

    # Ensure our dummy returned text
    assert "OK:Hello" in res.output

    # Verify captured kwargs include overrides and profile-derived fields filtered to Config
    args = _Capture.last_init_args or {}
    assert args.get("model") == "dummy-model"
    # Overrides applied
    assert args.get("quantization_bits") == 4
    assert args.get("quantization_method") == "bnb-nf4"


def test_cli_generate_env_profile_applied(monkeypatch: Any) -> None:
    import medvllm.tasks.text_generator as tg

    monkeypatch.setattr(tg, "LLM", DummyLLM, raising=True)

    # Set env var; no --deployment-profile flag
    monkeypatch.setenv("MEDVLLM_DEPLOYMENT_PROFILE", "gpu_8bit")

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "generate",
            "--text",
            "Hello",
            "--model",
            "dummy-model",
            "--json-meta",
        ],
    )  # type: ignore[arg-type]
    assert res.exit_code == 0, res.output

    args = _Capture.last_init_args or {}
    assert args.get("quantization_bits") == 8
    assert args.get("quantization_method") == "bnb-8bit"
