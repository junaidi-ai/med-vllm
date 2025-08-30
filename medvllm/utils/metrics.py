"""Utility functions for evaluating classification tasks.

Provides a thin wrapper over scikit-learn metrics to compute
accuracy, precision, recall, and F1 score in a consistent dict format.

Example:
    from medvllm.utils.metrics import compute_classification_metrics
    metrics = compute_classification_metrics(y_true, y_pred, average="macro")
    # metrics -> {"accuracy": float, "precision": float, "recall": float, "f1": float}
"""

from __future__ import annotations

from typing import Iterable, List, Optional


def compute_classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    labels: Optional[List] = None,
    average: str = "macro",
    zero_division: int | float = 0,
    y_score: Optional[Iterable] = None,
) -> dict:
    """Compute standard classification metrics.

    Args:
        y_true: Iterable of ground-truth labels.
        y_pred: Iterable of predicted labels.
        labels: Optional list of label names/ids to include for averaging.
                If None, scikit-learn infers labels from ``y_true``.
        average: Averaging strategy for multi-class. Common values: "macro",
                 "micro", "weighted". See sklearn docs for details.
        zero_division: Value to use when there is a zero division (e.g., when
                       no predicted samples for a label). Defaults to 0 to
                       avoid warnings in small fixtures.

    Returns:
        dict with keys: accuracy, precision, recall, f1 (floats in [0, 1]).
    """
    # Lazy import to avoid hard dependency at module import time
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    from sklearn.preprocessing import label_binarize

    acc = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=average,
        zero_division=zero_division,
    )

    result = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    # Optional ROC-AUC (multi-class one-vs-rest)
    if y_score is not None:
        try:
            # Determine unique classes from y_true (or provided labels)
            if labels is None:
                classes = sorted(set(list(y_true)))
            else:
                classes = list(labels)
            # Binarize ground truth
            y_true_bin = label_binarize(list(y_true), classes=classes)
            # Convert y_score to 2D array-like (n_samples, n_classes)
            # Accepts an iterable of per-sample sequences
            y_score_arr = list(y_score)
            # Compute macro-average one-vs-rest AUC if feasible
            auc_macro = roc_auc_score(y_true_bin, y_score_arr, average="macro", multi_class="ovr")
            auc_micro = roc_auc_score(y_true_bin, y_score_arr, average="micro", multi_class="ovr")
            result["auc_macro_ovr"] = float(auc_macro)
            result["auc_micro_ovr"] = float(auc_micro)
        except Exception:
            # Not computable (e.g., single class, degenerate scores). Leave AUC absent.
            result.setdefault("auc_macro_ovr", None)
            result.setdefault("auc_micro_ovr", None)

    return result
