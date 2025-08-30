import os
import re
from click.testing import CliRunner

from medvllm.cli import cli


def test_cli_conflict_flash_disabled_warning(monkeypatch):
    runner = CliRunner(mix_stderr=False)
    env = {"MEDVLLM_TEST_FAKE_ENGINE": "1"}
    # Invoke generate with conflicting flags
    result = runner.invoke(
        cli,
        [
            "inference",
            "generate",
            "--text",
            "hello",
            "--model",
            "fake-model",
            "--attention-impl",
            "flash",
            "--no-flash-attention",
        ],
        env=env,
    )
    assert result.exit_code == 0, result.output
    # Rich prints 'Warning:' prefix
    assert "Warning:" in result.output
    assert "--attention-impl flash requested" in result.output
