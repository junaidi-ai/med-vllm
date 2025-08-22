import json
from pathlib import Path

import pytest

from medvllm.tasks import NERProcessor
from medvllm.utils.ner_metrics import compute_ner_strict_metrics


def _load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@pytest.mark.integration
def test_ner_integration_on_fixture_dataset():
    path = Path("tests/fixtures/data/datasets/ner_dataset.jsonl")
    assert path.exists(), f"Missing fixture dataset at {path}"

    rows = _load_jsonl(path)

    # Use regex/gazetteer fallback with extended items enabled for better coverage
    from types import SimpleNamespace

    cfg = SimpleNamespace(ner_enable_extended_gazetteer=True)
    proc = NERProcessor(inference_pipeline=None, config=cfg)

    gold_docs = []
    pred_docs = []

    # Evaluate only on test split to mimic typical workflow
    test_rows = [r for r in rows if r.get("split") == "test"]
    assert test_rows, "Fixture must contain some test examples"

    for r in test_rows:
        text = r["text"]
        gold = r.get("entities", [])
        # Normalize to minimal fields used by metrics
        gold_min = [
            {"start": int(e["start"]), "end": int(e["end"]), "type": str(e["type"]).lower()}
            for e in gold
        ]
        res = proc.extract_entities(text)
        pred = [
            {"start": int(e["start"]), "end": int(e["end"]), "type": str(e["type"]).lower()}
            for e in res.entities
        ]
        gold_docs.append(gold_min)
        pred_docs.append(pred)

    metrics = compute_ner_strict_metrics(gold_docs, pred_docs)

    # Basic shape checks
    assert set(metrics.keys()) == {"micro", "per_type"}
    micro = metrics["micro"]
    for k in ("precision", "recall", "f1"):
        assert 0.0 <= micro[k] <= 1.0

    # Per-type values should be well-formed and in-range
    for t, m in metrics["per_type"].items():
        for k in ("precision", "recall", "f1"):
            assert 0.0 <= m[k] <= 1.0
