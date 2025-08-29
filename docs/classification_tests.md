# Classification Testing for Benchmarks

This guide explains how to run the optional text classification evaluation integrated into `benchmarks/benchmark_medical.py` using a simple majority-class baseline on a small fixture dataset.

## What it does
- Loads a CSV dataset with columns: `text,label,source,split`.
- Uses the training split to determine the majority label.
- Predicts the majority label for the test split.
- Computes accuracy, precision, recall, and F1 (macro) using scikit-learn.
- Saves metrics to JSON in the selected `--output-dir`.

## Requirements
- Python environment with the project installed or available on the PYTHONPATH.
- scikit-learn (installed via `requirements/requirements-test.txt`).

Example (recommended) using the project virtualenv:
```bash
# From repo root
./.venv/bin/pip install -r requirements/requirements-test.txt
```

## Running the evaluation
You can run the benchmark and the classification evaluation together. The dataset path defaults to the fixture CSV if not provided.

```bash
python3 benchmarks/benchmark_medical.py \
  --model biobert \
  --device cpu \
  --num-iterations 2 \
  --warmup-iterations 0 \
  --batch-sizes 1 \
  --seq-lengths 16 \
  --output-dir benchmarks/results \
  --test-accuracy
```

To explicitly specify a dataset CSV:
```bash
python3 benchmarks/benchmark_medical.py \
  --model biobert \
  --device cpu \
  --output-dir benchmarks/results \
  --test-accuracy \
  --dataset-csv tests/fixtures/data/datasets/text_classification_dataset.csv
```

## Output artifacts
Two JSON files are written (timestamps will differ):
- `<model>_benchmark_<ts>.json` — performance and memory stats
- `<model>_classification_metrics_<ts>.json` — classification metrics and dataset info

Example contents (`*_classification_metrics_*.json`):
```json
{
  "model_type": "biobert",
  "dataset_csv": ".../tests/fixtures/data/datasets/text_classification_dataset.csv",
  "baseline": "majority_class",
  "majority_label": "cardiac_emergency",
  "num_train": 3,
  "num_test": 2,
  "metrics": {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  },
  "timestamp": "2025-08-18T10:20:06.413295"
}
```

Note: The tiny fixture splits may yield zero metrics when the majority label in train does not appear in the test split. This is expected for the baseline.

## Debugging I/O
Use the optional flag to print write paths and size checks:
```bash
python3 benchmarks/benchmark_medical.py ... --test-accuracy --debug-io
```

## Troubleshooting
- Missing scikit-learn: install test requirements (`requirements/requirements-test.txt`).
- Dataset not found: provide `--dataset-csv` or ensure the default fixture exists.
- All-zero metrics: expected on tiny fixtures; try a larger dataset for more meaningful scores.
- CUDA unavailable: the script will automatically fall back to CPU if `--device cuda` is requested but CUDA is not available.
