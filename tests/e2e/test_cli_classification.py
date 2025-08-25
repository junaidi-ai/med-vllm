import json

import pytest

from ._helpers import run_cli


@pytest.mark.e2e
@pytest.mark.uat
def test_inference_classification_json_output():
    # Skip at runtime if transformers is unavailable or broken
    pytest.importorskip(
        "transformers", reason="transformers is not installed; skipping classification E2E"
    )
    code, out, err, _ = run_cli(
        [
            "inference",
            "classification",
            "--text",
            "This is very helpful!",
            "--json-out",
        ],
        timeout=60,
    )
    assert code == 0, f"classification CLI exited non-zero: {err}"
    data = json.loads(out)
    assert isinstance(data, dict)
    # Basic expected keys
    assert "label" in data
    assert "score" in data
