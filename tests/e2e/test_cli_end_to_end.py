import json
from pathlib import Path

import pytest

from ._helpers import run_cli


@pytest.mark.e2e
def test_examples_command_lists_common_commands():
    code, out, err, _ = run_cli(["examples"])
    assert code == 0, f"examples failed: {err}"
    assert "Quick examples" in out
    assert "inference ner" in out
    assert "model list" in out


@pytest.mark.e2e
def test_inference_ner_text_json_output():
    code, out, err, _ = run_cli(
        [
            "inference",
            "ner",
            "--text",
            "HTN on metformin",
            "--json-out",
        ]
    )
    assert code == 0, f"NER CLI exited non-zero: {err}"
    # Should be valid JSON
    data = json.loads(out)
    assert isinstance(data, dict)
    # Expect basic structure
    assert "entities" in data or "spans" in data


@pytest.mark.e2e
def test_inference_generate_from_stdin_and_file_output(tmp_path: Path):
    out_file = tmp_path / "gen.txt"
    prompt = "Explain hypertension in one sentence."
    code, out, err, _ = run_cli(
        [
            "inference",
            "generate",
            "--model",
            "gpt2",  # default tiny model path resolved by adapters/mocks
            "--output",
            str(out_file),
        ],
        input_text=prompt,
    )
    assert code == 0, f"generate exited non-zero: {err}"
    # Output file should exist and contain text
    content = out_file.read_text(encoding="utf-8").strip()
    assert len(content) > 0
    # CLI may also print metadata or nothing depending on flags; do not assert stdout content here


@pytest.mark.e2e
def test_model_list_runs():
    code, out, err, _ = run_cli(["model", "list"])
    # Should not crash even if registry is empty; the command should show something
    assert code == 0, f"model list failed: {err}"
    assert out.strip() != ""
