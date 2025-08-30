import re
from click.testing import CliRunner

from medvllm.cli import cli


def test_cli_pooling_flags_warning_when_disabled():
    runner = CliRunner()
    env = {"MEDVLLM_TEST_FAKE_ENGINE": "1"}
    # --no-memory-pooling with pool options should warn but still succeed
    result = runner.invoke(
        cli,
        [
            "inference",
            "generate",
            "--text",
            "hello",
            "--model",
            "fake-model",
            "--no-memory-pooling",
            "--pool-max-bytes",
            "1234",
            "--pool-device",
            "cpu",
        ],
        env=env,
    )
    assert result.exit_code == 0, result.output
    # Should include the specific pooling warning (allow wrapped whitespace)
    assert re.search(
        r"Memory pooling options \(--pool-\*\).*?disabled; options will be ignored\.",
        result.output,
        flags=re.IGNORECASE | re.S,
    )


def test_cli_pooling_flags_accepted_with_fake_engine():
    runner = CliRunner()
    env = {"MEDVLLM_TEST_FAKE_ENGINE": "1"}
    # Flags should be accepted and not crash
    result = runner.invoke(
        cli,
        [
            "inference",
            "generate",
            "--text",
            "hello",
            "--model",
            "fake-model",
            "--memory-pooling",
            "--pool-max-bytes",
            "8192",
            "--pool-device",
            "auto",
        ],
        env=env,
    )
    assert result.exit_code == 0, result.output
    # No pooling-specific warning expected in this valid combination
    assert (
        "Memory pooling options (--pool-*) were provided but pooling was disabled; options will be ignored."
        not in result.output
    )
