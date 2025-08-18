import os
from collections import Counter

import pytest

# Skip entirely if scikit-learn is unavailable
pytest.importorskip("sklearn")

# Skip module by default unless explicitly enabled
if os.environ.get("MEDVLLM_RUN_SLOW", "").lower() not in {"1", "true", "yes"}:
    pytest.skip(
        "Slow tests disabled by default; set MEDVLLM_RUN_SLOW=1 to enable",
        allow_module_level=True,
    )


@pytest.mark.slow
@pytest.mark.integration
def test_pubmed_qa_small_sample_majority_baseline():
    """Optional slow test: load a tiny PubMedQA sample and compute metrics.

    Notes:
    - Uses HF datasets; requires internet on first run unless cached.
    - Designed to be lightweight via split slicing (train[:50]).
    - Computes a simple majority-class baseline over 'final_decision'.
    - Skips gracefully if datasets is unavailable or network/cache fails.
    """
    # Allow users/CI to disable networked tests explicitly
    if os.environ.get("MEDVLLM_SKIP_HF_DATASETS", "").lower() in {"1", "true", "yes"}:
        pytest.skip("Skipping HF datasets per MEDVLLM_SKIP_HF_DATASETS env var")

    try:
        from datasets import load_dataset  # lazy import to avoid hard dep in fast paths
    except Exception as e:
        pytest.skip(f"datasets library not available: {e}")

    try:
        # PubMedQA labeled split provides 'final_decision' in {yes,no,maybe}
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train[:50]")
    except Exception as e:
        pytest.skip(f"Could not load PubMedQA small sample (likely offline): {e}")

    # Extract labels; ensure non-empty
    labels = list(ds["final_decision"]) if "final_decision" in ds.column_names else []
    if not labels:
        pytest.skip("PubMedQA sample missing 'final_decision' labels; dataset schema changed?")

    # Majority-class baseline
    majority = Counter(labels).most_common(1)[0][0]
    y_true = labels
    y_pred = [majority] * len(labels)

    from medvllm.utils.metrics import compute_classification_metrics

    metrics = compute_classification_metrics(y_true, y_pred, average="macro")

    # Basic validity checks
    for k in ("accuracy", "precision", "recall", "f1"):
        assert k in metrics
        assert 0.0 <= metrics[k] <= 1.0

    # Do not assert a fixed value (keeps test robust across dataset changes)
