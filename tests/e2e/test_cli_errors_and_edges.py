import json
import os
from pathlib import Path

import pytest

from ._helpers import run_cli


@pytest.mark.e2e
@pytest.mark.parametrize(
    "args",
    [
        ["inference", "ner"],
        [
            "inference",
            "generate",
            "--model",
            "gpt2",  # model validated first; read_input then errors due to no input
        ],
    ],
)
def test_error_no_input_provided(args):
    code, out, err, _ = run_cli(args)
    # Click UsageError yields non-zero and prints guidance in stderr
    assert code != 0
    combined = out + err
    assert "No input provided" in combined
    assert "--text" in combined or "--input" in combined


@pytest.mark.e2e
def test_error_both_text_and_input_for_ner(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("sample", encoding="utf-8")
    code, out, err, _ = run_cli(
        [
            "inference",
            "ner",
            "--text",
            "abc",
            "--input",
            str(f),
        ]
    )
    assert code != 0
    combined = out + err
    assert "Provide only one of --text or --input" in combined


@pytest.mark.e2e
def test_pdf_requires_pypdf_or_fails_gracefully(tmp_path: Path):
    pdf = tmp_path / "empty.pdf"
    # Create a minimal PDF header to mimic a file; extraction will fail or be empty
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    code, out, err, _ = run_cli(
        [
            "inference",
            "ner",
            "--input",
            str(pdf),
            "--input-format",
            "pdf",
            "--json-out",
        ]
    )
    if code == 0:
        # Some environments may have pypdf and tolerate empty PDF; ensure JSON
        data = json.loads(out)
        assert isinstance(data, dict)
    else:
        # Expect a helpful message (either missing pypdf or no extractable text)
        combined = out + err
        assert ("pypdf" in combined.lower()) or ("no extractable text" in combined.lower())
