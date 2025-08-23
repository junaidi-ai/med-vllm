# Medical Model Benchmarks - CPU (local time: 2025-08-23)

## Setup
- Device: `cpu`, Precision: `fp32`
- Batch size: 1, Seq length: 128
- Env: local virtualenv `.venv` (for test deps)
- Notes:
  - `Flash Attention` not installed (CPU fallback OK)
  - NVIDIA driver outdated; CPU-only
  - For scripts importing `medvllm`, used `PYTHONPATH=.`

## Results
- BioBERT benchmark (`benchmark_results_cpu_smoke/biobert_benchmark_1755885392.json`)
  - Avg latency: 361.02 ms
  - Throughput: 354.55 tokens/sec
  - Memory (MB): RSS Δ 318.25, VMS Δ 1.00
- ClinicalBERT benchmark (`benchmark_results_cpu_smoke/clinicalbert_benchmark_1755884192.json`)
  - Avg latency: 131.84 ms
  - Throughput: 970.86 tokens/sec
  - Memory (MB): RSS Δ 0.00, VMS Δ 0.00
- NER benchmark (`benchmark_results_cpu_smoke/ner_benchmark_results.json`)
  - Num examples: 2
  - Avg run latency: 0.000182 s
  - Throughput: 10,964.99 examples/sec
  - Strict metrics (micro): P/R/F1 = 0.0/0.0/0.0
- Ontology linking cache (stdout only)
  - Extracted entities: 180
  - Run1 0.0003s | hits=174, misses=6, size=6
  - Run2 0.0001s | hits=354, misses=6, size=6
  - Run3 0.0001s | hits=534, misses=6, size=6
  - Interpretation: first run populates ~6 entries; subsequent runs are near-100% cache hits
- Classification baseline metrics (`benchmark_results_cpu_smoke/biobert_classification_metrics_1755885392.json`)
  - Dataset: `tests/fixtures/data/datasets/text_classification_dataset.csv`
  - Baseline: `majority_class` (label: `cardiac_emergency`)
  - Train/Test: 3 / 2
  - Accuracy/Precision/Recall/F1: 0.0 / 0.0 / 0.0 / 0.0

## Artifacts
- Benchmarks
  - `benchmark_results_cpu_smoke/biobert_benchmark_1755885392.json`
  - `benchmark_results_cpu_smoke/clinicalbert_benchmark_1755884192.json`
- NER
  - `benchmark_results_cpu_smoke/ner_benchmark_results.json`
- Classification
  - `benchmark_results_cpu_smoke/biobert_classification_metrics_1755885392.json`

## Repro commands
```bash
# Classification (uses venv)
.venv/bin/python benchmarks/benchmark_medical.py \
  --model biobert --device cpu --precision fp32 \
  --batch-sizes 1 --seq-lengths 128 \
  --num-iterations 1 --warmup-iterations 0 \
  --output-dir benchmark_results_cpu_smoke --test-accuracy

# NER (needs PYTHONPATH for local package imports)
PYTHONPATH=. python3 benchmarks/benchmark_ner.py \
  --dataset tests/fixtures/data/datasets/ner_dataset.jsonl \
  --output benchmark_results_cpu_smoke/ner_benchmark_results.json \
  --warmup 1 --runs 3

# Ontology linking (stdout shows cache stats)
PYTHONPATH=. python3 benchmarks/benchmark_linking.py \
  --paragraphs 30 --runs 3 --ontology RXNORM
```

## Adapter Benchmarks (Extended Grid, CPU)
- __Grid__: batch sizes [1, 2, 4]; seq lengths [64, 128, 256]; warmup 5; iterations 30; device cpu; fp32
- __Notes__: Flash Attention not installed (falls back to standard attention). Adapters handle tokenizer extensions automatically.

- __BioBERT__ (`monologg/biobert_v1.1_pubmed`)
  - JSON: `benchmark_results_cpu_smoke/biobert_adapter_benchmark_20250823_095012.json`
  - Log: `benchmark_results_cpu_smoke/biobert_extended.log`
  - Highlight: best throughput ≈ 1,139 tokens/s at batch=1, seq=64

- __ClinicalBERT__ (`emilyalsentzer/Bio_ClinicalBERT`)
  - JSON: `benchmark_results_cpu_smoke/clinicalbert_adapter_benchmark_20250823_095103.json`
  - Log: `benchmark_results_cpu_smoke/clinicalbert_extended.log`
  - Highlight: best throughput ≈ 1,390 tokens/s at batch=4, seq=256
