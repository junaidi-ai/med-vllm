import json
from pathlib import Path

import pytest

from medvllm.utils.ner_metrics import compute_ner_strict_metrics, compute_ner_metrics


@pytest.mark.unit
def test_compute_ner_strict_metrics_known_values_micro_and_per_type():
    # One document with two gold entities
    y_true = [
        [
            {"start": 0, "end": 3, "type": "DISEASE"},
            {"start": 5, "end": 9, "type": "MEDICATION"},
        ]
    ]
    # Predictions: correct disease span/type; wrong medication span (strict mismatch) and an extra entity
    y_pred = [
        [
            {"start": 0, "end": 3, "type": "disease"},  # TP
            {"start": 5, "end": 8, "type": "MEDICATION"},  # FP (strict span mismatch vs gold)
            {"start": 20, "end": 25, "type": "finding"},  # FP extra
        ]
    ]

    m = compute_ner_strict_metrics(y_true, y_pred)

    # Micro counts: TP=1, FP=2, FN=1  -> P=1/3, R=1/2, F1=2/5
    micro = m["micro"]
    assert micro["tp"] == 1
    assert micro["fp"] == 2
    assert micro["fn"] == 1
    assert pytest.approx(micro["precision"], rel=1e-9) == 1.0 / 3.0
    assert pytest.approx(micro["recall"], rel=1e-9) == 1.0 / 2.0
    assert pytest.approx(micro["f1"], rel=1e-9) == 2.0 * (1.0 / 3.0) * (1.0 / 2.0) / (
        (1.0 / 3.0) + (1.0 / 2.0)
    )

    # Per-type expectations (types lowercased)
    per = m["per_type"]
    # disease: tp=1, fp=0, fn=0 -> P=R=F1=1
    d = per.get("disease")
    assert d and d["tp"] == 1 and d["fp"] == 0 and d["fn"] == 0
    assert d["precision"] == 1.0 and d["recall"] == 1.0 and d["f1"] == 1.0
    # medication: tp=0, fp=1 (mismatch), fn=1 -> P=0, R=0, F1=0
    md = per.get("medication")
    assert md and md["tp"] == 0 and md["fp"] == 1 and md["fn"] == 1
    assert md["precision"] == 0.0 and md["recall"] == 0.0 and md["f1"] == 0.0


@pytest.mark.unit
def test_compute_ner_strict_metrics_accepts_flat_lists_and_doc_mismatch_error():
    # Flat lists (no doc partition) are accepted
    y_true_flat = [
        {"start": 0, "end": 3, "type": "disease"},
        {"start": 5, "end": 9, "type": "medication"},
    ]
    y_pred_flat = [
        {"start": 0, "end": 3, "type": "disease"},
    ]
    m = compute_ner_strict_metrics(y_true_flat, y_pred_flat)
    assert "micro" in m and "per_type" in m

    # Mismatched number of docs should raise
    y_true_docs = [y_true_flat, []]
    y_pred_docs = [y_pred_flat]
    with pytest.raises(ValueError):
        compute_ner_strict_metrics(y_true_docs, y_pred_docs)


@pytest.mark.unit
def test_compute_ner_overlap_iou_matching_threshold():
    # Gold single entity [0, 10], disease
    y_true = [[{"start": 0, "end": 10, "type": "disease"}]]
    # Predicted partially overlapping entity [0, 5], IoU = 5 / (10 + 5 - 5) = 0.5
    y_pred = [[{"start": 0, "end": 5, "type": "disease"}]]

    # Strict should not match
    m_strict = compute_ner_metrics(y_true, y_pred, match="strict")
    assert m_strict["micro"]["tp"] == 0 and m_strict["micro"]["fn"] == 1

    # Overlap with threshold 0.5 should match
    m_overlap = compute_ner_metrics(y_true, y_pred, match="overlap", iou_threshold=0.5)
    assert m_overlap["micro"]["tp"] == 1 and m_overlap["micro"]["fn"] == 0
    assert pytest.approx(m_overlap["micro"]["precision"]) == 1.0
    assert pytest.approx(m_overlap["micro"]["recall"]) == 1.0
    assert pytest.approx(m_overlap["micro"]["f1"]) == 1.0

    # Higher threshold should not match
    m_overlap_high = compute_ner_metrics(y_true, y_pred, match="overlap", iou_threshold=0.75)
    assert m_overlap_high["micro"]["tp"] == 0 and m_overlap_high["micro"]["fn"] == 1


@pytest.mark.unit
def test_compute_ner_macro_averaging_known_values():
    # Two docs, two types: 'a' and 'b'.
    # Doc1 gold: A (TP), B (FN) ; Doc2 gold: A (FN) and pred A is FP (strict mismatch)
    y_true = [
        [
            {"start": 0, "end": 3, "type": "A"},  # will be TP
            {"start": 5, "end": 8, "type": "B"},  # will be FN
        ],
        [
            {"start": 0, "end": 3, "type": "A"},  # will be FN
        ],
    ]
    y_pred = [
        [
            {"start": 0, "end": 3, "type": "a"},  # TP for A
        ],
        [
            {"start": 0, "end": 2, "type": "A"},  # FP for A (strict mismatch)
        ],
    ]

    m = compute_ner_metrics(y_true, y_pred, match="strict")
    micro = m["micro"]
    macro = m["macro"]
    per = m["per_type"]

    # Per-type A: tp=1, fp=1, fn=1 -> P=0.5, R=0.5, F1=0.5
    A = per["a"]
    assert A["tp"] == 1 and A["fp"] == 1 and A["fn"] == 1
    assert pytest.approx(A["precision"]) == 0.5
    assert pytest.approx(A["recall"]) == 0.5
    assert pytest.approx(A["f1"]) == 0.5

    # Per-type B: tp=0, fp=0, fn=1 -> P=0, R=0, F1=0
    B = per["b"]
    assert B["tp"] == 0 and B["fp"] == 0 and B["fn"] == 1
    assert B["precision"] == 0.0 and B["recall"] == 0.0 and B["f1"] == 0.0

    # Macro averages are the mean of per-type scores: (0.5 + 0.0)/2 = 0.25
    assert pytest.approx(macro["precision"]) == 0.25
    assert pytest.approx(macro["recall"]) == 0.25
    assert pytest.approx(macro["f1"]) == 0.25

    # Micro across all: tp=1, fp=1, fn=2 -> P=0.5, R=1/3, F1=0.4
    assert pytest.approx(micro["precision"]) == 0.5
    assert pytest.approx(micro["recall"]) == pytest.approx(1.0 / 3.0)
    assert pytest.approx(micro["f1"]) == pytest.approx(0.4)
