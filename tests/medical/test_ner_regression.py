import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from medvllm.tasks import NERProcessor
from medvllm.utils.ner_metrics import compute_ner_metrics


@pytest.mark.regression
def test_ner_micro_f1_regression_guard_on_fixture():
    """
    Guardrail: Ensure micro F1 on the fixture dataset does not catastrophically drop.
    Uses strict span+type matching and a very conservative threshold.
    """
    path = Path("tests/fixtures/data/datasets/ner_dataset.jsonl")
    assert path.exists(), f"Missing fixture dataset at {path}"

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    test_rows = [r for r in rows if str(r.get("split", "")).lower() == "test"] or rows
    assert test_rows, "Fixture must contain some test examples"

    cfg = SimpleNamespace(ner_enable_extended_gazetteer=True)
    proc = NERProcessor(inference_pipeline=None, config=cfg)

    gold_docs = []
    pred_docs = []
    for r in test_rows:
        text = r.get("text", "")
        gold = r.get("entities", [])
        gold_docs.append(
            [
                {"start": int(e["start"]), "end": int(e["end"]), "type": str(e["type"]).lower()}
                for e in gold
            ]
        )
        res = proc.extract_entities(text)
        pred_docs.append(
            [
                {"start": int(e["start"]), "end": int(e["end"]), "type": str(e["type"]).lower()}
                for e in res.entities
            ]
        )

    strict = compute_ner_metrics(gold_docs, pred_docs, match="strict")
    strict_f1 = strict["micro"]["f1"]
    assert 0.0 <= strict_f1 <= 1.0  # sanity

    # Use lenient overlap IoU>=0.5 as regression guard; conservative threshold
    overlap = compute_ner_metrics(gold_docs, pred_docs, match="overlap", iou_threshold=0.5)
    overlap_f1 = overlap["micro"]["f1"]
    assert 0.0 <= overlap_f1 <= 1.0
    assert overlap_f1 >= 0.10, f"NER overlap micro F1 too low on fixture: {overlap_f1:.4f} < 0.10"

    # Sanity: overlap should be >= strict
    assert overlap_f1 + 1e-9 >= strict_f1
