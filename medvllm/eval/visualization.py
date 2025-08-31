from __future__ import annotations

from typing import List, Optional, Sequence

import io


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover - optional
        raise RuntimeError("matplotlib is required for plotting; please install it") from e
    return plt


def plot_confusion_matrix_png(
    cm: List[List[int]],
    labels: List[str],
    *,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    out_path: Optional[str] = None,
) -> bytes:
    """Render confusion matrix to PNG bytes; optionally save to out_path."""
    import numpy as np  # lightweight dep

    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.imshow(np.array(cm), interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate counts
    thresh = (max(max(row) for row in cm) or 1) / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i][j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = buf.getvalue()
    if out_path:
        with open(out_path, "wb") as f:
            f.write(data)
    plt.close(fig)
    return data


def plot_roc_curve_png(
    y_true: Sequence[int | str],
    y_score: Sequence[float] | Sequence[Sequence[float]],
    *,
    out_path: Optional[str] = None,
    title: str = "ROC Curve",
) -> bytes:
    """Plot ROC. Supports binary (y_score: probs for positive class) and
    multiclass (y_score: per-class probabilities). Uses sklearn when available.
    """
    try:
        import numpy as np
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
    except Exception as e:  # pragma: no cover - optional
        raise RuntimeError("scikit-learn is required for ROC plotting") from e

    plt = _require_matplotlib()

    y_true_str = [str(x) for x in y_true]
    classes = sorted(set(y_true_str))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)

    if len(classes) == 2 and isinstance(y_score[0], (int, float)):
        # binary
        y_bin = [1 if y == classes[1] else 0 for y in y_true_str]
        fpr, tpr, _ = roc_curve(y_bin, y_score)  # type: ignore[arg-type]
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    else:
        # multiclass one-vs-rest
        y_bin = label_binarize(y_true_str, classes=classes)
        scores = np.array(y_score)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    data = buf.getvalue()
    if out_path:
        with open(out_path, "wb") as f:
            f.write(data)
    plt.close(fig)
    return data


def plot_pr_curve_png(
    y_true: Sequence[int | str],
    y_score: Sequence[float] | Sequence[Sequence[float]],
    *,
    out_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
) -> bytes:
    try:
        import numpy as np
        from sklearn.metrics import precision_recall_curve, auc
        from sklearn.preprocessing import label_binarize
    except Exception as e:  # pragma: no cover - optional
        raise RuntimeError("scikit-learn is required for PR plotting") from e

    plt = _require_matplotlib()

    y_true_str = [str(x) for x in y_true]
    classes = sorted(set(y_true_str))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)

    if len(classes) == 2 and isinstance(y_score[0], (int, float)):
        y_bin = [1 if y == classes[1] else 0 for y in y_true_str]
        prec, rec, _ = precision_recall_curve(y_bin, y_score)  # type: ignore[arg-type]
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color="darkgreen", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    else:
        y_bin = label_binarize(y_true_str, classes=classes)
        scores = np.array(y_score)
        for i, cls in enumerate(classes):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], scores[:, i])
            pr_auc = auc(rec, prec)
            ax.plot(rec, prec, lw=2, label=f"{cls} (AUC = {pr_auc:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    data = buf.getvalue()
    if out_path:
        with open(out_path, "wb") as f:
            f.write(data)
    plt.close(fig)
    return data
