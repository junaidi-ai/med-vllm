from __future__ import annotations

from typing import Dict, Any
import json

DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # Task domains and suggested floors; adjust per clinical policy
    # Keys map to metric names in ClassificationMetrics
    "classification.general": {
        "accuracy": 0.95,
        "f1_macro": 0.92,
    },
    "classification.risk_stratification": {
        "accuracy": 0.97,
        "f1_macro": 0.95,
    },
    "classification.alerting": {
        "recall_macro": 0.96,  # prioritize sensitivity
        "precision_macro": 0.90,
    },
    "classification.smoke_cpu": {
        # Relaxed thresholds for CPU-only smoke validations (e.g., dynamic int8)
        # Intent: catch gross regressions while avoiding false negatives due to
        # known small accuracy drops from post-training dynamic quantization.
        "accuracy": 0.88,
        "f1_macro": 0.88,
    },
    "ner.medical": {
        # For future extension; current CLI focuses on classification
        # Example thresholds if needed later
        # "f1_micro": 0.90,
    },
}


def load_thresholds_from_file(path: str) -> Dict[str, Any]:
    """Load thresholds from a file. Supports JSON and YAML (if PyYAML installed).

    Returns a dict; for classification we generally expect either:
      - flat: {"accuracy": 0.95, "f1_macro": 0.92}
      - namespaced: {"classification.general": {...}}
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Try JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try YAML if available
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
        raise ValueError("YAML thresholds must be a mapping")
    except Exception as e:
        raise ValueError(
            f"Unsupported thresholds file format for {path}; provide JSON or YAML. Error: {e}"
        ) from e
