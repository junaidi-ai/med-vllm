import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# Skip tests if scikit-learn is not available
pytest.importorskip("sklearn")
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from medvllm.utils.metrics import compute_classification_metrics


def test_compute_classification_metrics_known_values():
    # Small deterministic example
    y_true = [0, 1, 2, 2]
    y_pred = [0, 0, 2, 1]

    metrics = compute_classification_metrics(y_true, y_pred, average="macro")

    acc_exp = accuracy_score(y_true, y_pred)
    p_exp, r_exp, f1_exp, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    assert pytest.approx(metrics["accuracy"], rel=1e-9, abs=1e-12) == float(acc_exp)
    assert pytest.approx(metrics["precision"], rel=1e-9, abs=1e-12) == float(p_exp)
    assert pytest.approx(metrics["recall"], rel=1e-9, abs=1e-12) == float(r_exp)
    assert pytest.approx(metrics["f1"], rel=1e-9, abs=1e-12) == float(f1_exp)


def _load_text_classification_fixture():
    path = Path("tests/fixtures/data/datasets/text_classification_dataset.csv")
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _majority_label_train(rows):
    # Use deterministic tie-break: lexicographically smallest label among max-count labels
    train_labels = [r["label"] for r in rows if r.get("split") == "train"]
    counts = Counter(train_labels)
    if not counts:
        raise RuntimeError("No train rows in fixture dataset")
    max_count = max(counts.values())
    candidates = sorted([lbl for lbl, cnt in counts.items() if cnt == max_count])
    return candidates[0]


def test_classification_metrics_on_fixture_majority_baseline():
    rows = _load_text_classification_fixture()

    majority = _majority_label_train(rows)

    y_true = [r["label"] for r in rows if r.get("split") == "test"]
    y_pred = [majority for _ in y_true]

    # Sanity checks
    assert len(y_true) > 0, "Fixture must contain test split rows"

    metrics = compute_classification_metrics(y_true, y_pred, average="macro")

    # Keys and ranges
    for k in ("accuracy", "precision", "recall", "f1"):
        assert k in metrics
        assert 0.0 <= metrics[k] <= 1.0

    # Match sklearn's accuracy
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))

    # Note: With the current fixture, majority label (from train) does not
    # appear in the test set, so accuracy is expected to be 0.0. If the
    # fixture changes, this assertion can be relaxed or updated.
    # Keep it informational for now without hard-coding 0.0 to avoid brittleness.
