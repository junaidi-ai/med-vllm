# Clinical Validation Workflow

This guide explains how to run accuracy validation, apply thresholds, generate plots, and integrate with existing benchmarks in Med‑vLLM.

## 1) Run Quantization Accuracy Benchmark with Validation

Use a task‑specific fine‑tuned model and realistic thresholds. Example (SST‑2):

```bash
python benchmarks/benchmark_quantization_accuracy.py \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  --dataset-csv tests/fixtures/data/datasets/text_classification_dataset.csv \
  --limit 256 \
  --validate \
  --out-dir reports \
  --prefix quant_acc_sst2 \
  --thresholds-preset classification.general \
  --threshold accuracy=0.85 --threshold f1_macro=0.80
```

Outputs:
- JSON summary to stdout
- Validation report at `reports/<prefix>_report.json` with metrics, McNemar test, and threshold pass/fail

Notes:
- `bnb-8bit` path requires `bitsandbytes` and CUDA; otherwise it is skipped gracefully.
- Ensure dataset labels align with the model’s task/domain to get meaningful metrics.

## 2) Thresholds: Presets, Files, and Inline Overrides

- Preset: `--thresholds-preset classification.general`
- From file (JSON/YAML): `--thresholds-file thresholds.yaml`
  - Example YAML:
    ```yaml
    accuracy: 0.90
    f1_macro: 0.88
    ```
- Inline overrides (highest precedence): `--threshold accuracy=0.90 --threshold f1_macro=0.88`

Precedence: inline > file > preset.

## 3) Run Validation CLI on a Predictions CSV

Prepare a CSV with columns for the true labels and two prediction columns.
Example command:

```bash
python -m medvllm.cli.validation_commands classification-validate \
  --csv path/to/your.csv \
  --y-true-col y_true \
  --y-pred-a-col y_pred_baseline \
  --y-pred-b-col y_pred_opt \
  --out-dir reports/validate_run \
  --thresholds-preset classification.general \
  --threshold accuracy=0.90
```

Outputs in `--out-dir`:
- JSON report (metrics, McNemar, thresholds)
- Confusion matrix PNG; ROC/PR plots if applicable

## 4) Benchmarks Integration

- Quantization validation: `benchmarks/benchmark_quantization_accuracy.py`
- Original vs Med‑vLLM: `benchmarks/compare_with_original.py`
  - Handles missing `predict()` gracefully; records timings and memory usage.

## 5) Environment Dependencies

Install medical validation extras:

```bash
pip install -r requirements/requirements-medical.txt
```

Optional for bnb‑8bit:

```bash
pip install -U bitsandbytes
```

## 6) Interpreting Validation Reports

Each report contains:
- Classification metrics (accuracy, precision/recall/F1 macro & micro)
- Confusion matrix and label order used
- McNemar test (b, c, statistic, p‑value) for paired predictions
- Thresholds applied and per‑metric pass/fail for baseline and optimized models

## 7) Tips for Clinical‑Grade Runs

- Use clinically relevant, fine‑tuned models aligned with your dataset labels.
- Set thresholds that reflect acceptance criteria (preset/file/inline).
- Keep raw prediction CSVs for auditability alongside JSON reports and plots.
- Version control the thresholds file used for each validation.

## 8) CPU Smoke Preset

For CPU‑only smoke tests with dynamic int8 quantization, use the relaxed preset:

```bash
--thresholds-preset classification.smoke_cpu
```

This preset sets both `accuracy` and `f1_macro` floors to 0.88 to reduce false alarms from the small, expected drop with post‑training dynamic quantization, while still catching material regressions.

## 9) Clinical Sign‑off Metadata

You can attach clinical sign‑off details to validation reports via inline flags or a file:

Inline example:

```bash
python -m medvllm.cli.validation_commands classification-validate \
  --csv path/to/preds.csv \
  --y-true-col y_true --y-pred-a-col baseline --y-pred-b-col optimized \
  --out-dir reports/validate_demo \
  --thresholds-preset classification.general \
  --signoff reviewer="Dr. Lee" --signoff approved=true --signoff notes="Meets criteria"
```

File-based (JSON/YAML) example:

```bash
python -m medvllm.cli.validation_commands classification-validate \
  --csv path/to/preds.csv \
  --y-true-col y_true --y-pred-a-col baseline --y-pred-b-col optimized \
  --out-dir reports/validate_demo \
  --thresholds-preset classification.general \
  --signoff-file docs/examples/signoff_example.json
```

Reports include a `signoff` block with fields (e.g., reviewer, date, approved, notes). Archive signed JSONs and plots under a dated directory, e.g., `reports/2025-08-31/<run_id>/` for auditability.

## 10) CI Gating on Thresholds

Use `--fail-on-thresholds` to exit non‑zero when the optimized model fails threshold checks:

```bash
python -m medvllm.cli.validation_commands classification-validate \
  --csv path/to/preds.csv \
  --y-true-col y_true --y-pred-a-col baseline --y-pred-b-col optimized \
  --thresholds-preset classification.general \
  --fail-on-thresholds
```

Benchmark variant (also supports sign‑off and scores export):

```bash
python benchmarks/benchmark_quantization_accuracy.py \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  --dataset-csv reports/sst2_validation.csv --limit 128 --validate \
  --thresholds-preset classification.general \
  --save-probs-json \
  --signoff reviewer="Dr. Lee" --signoff approved=true \
  --out-dir reports --prefix ci_val --fail-on-thresholds
```

The repository CI includes a CPU smoke validation step using `classification.smoke_cpu` that fails the build if thresholds are not met.

## 11) ROC/PR from Benchmark Probabilities

`benchmark_quantization_accuracy.py` can emit `<prefix>_scores.json` (with `--save-probs-json`) containing:

- `y_true`: true labels
- `scores_a`: baseline probabilities (class‑1 for binary)
- `scores_b_dynamic` / `scores_b_bnb`: optimized probabilities

Use this with the CLI to render ROC/PR plots:

```bash
python -m medvllm.cli.validation_commands classification-validate \
  --csv path/to/preds.csv \
  --y-true-col y_true --y-pred-a-col baseline --y-pred-b-col optimized \
  --out-dir reports/validate_demo \
  --thresholds-preset classification.general \
  --roc-scores-json reports/ci_val_scores.json
```
