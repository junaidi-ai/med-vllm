from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math

# Keep hard deps minimal; rely on stdlib when possible and import optional libs lazily


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    confusion_matrix: List[List[int]]
    labels: List[str]


def _unique_labels(y_true: Sequence, y_pred: Sequence) -> List[str]:
    labels = sorted({str(x) for x in y_true} | {str(x) for x in y_pred})
    return labels


def _confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: List[str]) -> List[List[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        i = idx[str(t)]
        j = idx[str(p)]
        m[i][j] += 1
    return m


def _safe_div(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def compute_classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    labels: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """Compute core metrics using only stdlib.

    Returns macro/micro precision/recall/F1 and accuracy + confusion matrix.
    """
    if labels is None:
        labels = _unique_labels(y_true, y_pred)
    cm = _confusion_matrix(y_true, y_pred, labels)

    # Totals
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(labels)))
    accuracy = _safe_div(correct, total)

    # Per-class
    precs: List[float] = []
    recs: List[float] = []
    f1s: List[float] = []

    # Micro counts
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0

    for i, _ in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)

        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0

        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    precision_macro = sum(precs) / len(precs) if precs else 0.0
    recall_macro = sum(recs) / len(recs) if recs else 0.0
    f1_macro = sum(f1s) / len(f1s) if f1s else 0.0

    precision_micro = _safe_div(tp_sum, tp_sum + fp_sum)
    recall_micro = _safe_div(tp_sum, tp_sum + fn_sum)
    f1_micro = (
        _safe_div(2 * precision_micro * recall_micro, precision_micro + recall_micro)
        if (precision_micro + recall_micro)
        else 0.0
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_micro=precision_micro,
        recall_micro=recall_micro,
        f1_micro=f1_micro,
        confusion_matrix=cm,
        labels=labels,
    )


@dataclass
class McNemarResult:
    b: int
    c: int
    statistic: float
    pvalue: float
    exact: bool


def mcnemar_test_equivalence(
    y_true: Sequence,
    y_pred_a: Sequence,
    y_pred_b: Sequence,
    *,
    exact_threshold: int = 25,
) -> McNemarResult:
    """Run McNemar's test on paired predictions A vs B.

    Uses scipy when available. Otherwise, uses chi-squared with continuity correction.
    """
    # Build discordant counts
    b = 0  # A wrong, B correct
    c = 0  # A correct, B wrong
    for t, a, bpred in zip(y_true, y_pred_a, y_pred_b):
        correct_a = str(a) == str(t)
        correct_b = str(bpred) == str(t)
        if (not correct_a) and correct_b:
            b += 1
        elif correct_a and (not correct_b):
            c += 1

    n = b + c
    if n == 0:
        # No discordant pairs: identical correctness; p=1.0 by definition
        return McNemarResult(b=b, c=c, statistic=0.0, pvalue=1.0, exact=False)

    # Try scipy exact if small n
    try:
        from scipy.stats import mcnemar  # type: ignore

        table = [[0, c], [b, 0]]
        exact = n <= exact_threshold
        res = mcnemar(table, exact=exact, correction=not exact)
        return McNemarResult(
            b=b, c=c, statistic=float(res.statistic), pvalue=float(res.pvalue), exact=exact
        )
    except Exception:
        # Fallback: chi-squared with continuity correction
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        # Two-sided approximate p-value via chi2 with df=1
        try:
            from math import exp

            # Survival function approx for chi2(1): p = exp(-stat/2)
            p = math.exp(-stat / 2.0)
        except Exception:
            p = 1.0
        return McNemarResult(b=b, c=c, statistic=float(stat), pvalue=float(p), exact=False)


def threshold_check(
    metrics: ClassificationMetrics,
    thresholds: Dict[str, float],
) -> Dict[str, bool]:
    """Compare metrics to threshold dict: keys can be 'accuracy', 'f1_macro', etc.

    Returns a dict of pass/fail per key.
    """
    results: Dict[str, bool] = {}
    for key, th in thresholds.items():
        if not hasattr(metrics, key):
            continue
        val = getattr(metrics, key)
        try:
            results[key] = bool(val >= th)
        except Exception:
            results[key] = False
    return results
