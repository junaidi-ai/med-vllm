import time
from pathlib import Path

import pytest

from ._helpers import run_cli


@pytest.mark.e2e
@pytest.mark.performance
def test_ner_large_input_completes_quickly():
    # Construct a moderately large synthetic note (~50k chars) but simple content
    sentence = "Patient with hypertension on metformin. "
    text = sentence * 1200  # ~50k chars

    start = time.perf_counter()
    code, out, err, _ = run_cli(
        [
            "inference",
            "ner",
            "--text",
            text,
            "--json-out",
        ],
        timeout=45,
    )
    dur = time.perf_counter() - start

    assert code == 0, f"NER failed on large input: {err}"
    # Be generous for CI, but catch pathological slowness
    assert dur < 20.0, f"NER on large input took too long: {dur:.2f}s"


@pytest.mark.e2e
@pytest.mark.uat
def test_generate_minimal_happy_path_ua():
    # UAT-style: ensure basic happy path does not crash and yields some text
    code, out, err, _ = run_cli(
        [
            "inference",
            "generate",
            "--text",
            "Summarize diabetes.",
            "--model",
            "gpt2",
            "--strategy",
            "greedy",
            "--max-length",
            "30",
        ]
    )
    assert code == 0, f"generate exited non-zero: {err}"
    # Some text printed
    assert out.strip() != ""
