# Benchmarks

This folder contains scripts and utilities to benchmark adapters and training.

All benchmark outputs are consolidated under:
- `benchmarks/results/`

Legacy/scattered output directories are now gitignored:
- `.benchmarks/`, `.benchmarks_train_tmp/`
- `benchmark_results/`, `benchmark_results_cpu_smoke/`, `benchmark_results_gpu_smoke/`

## Quick Starts

- CPU adapter smoke (BioBERT):
  ```bash
  python benchmarks/benchmark_medical.py \
    --model biobert --batch-sizes 1 --seq-lengths 64 \
    --num-iterations 1 --warmup-iterations 0 \
    --precision fp32 --device cpu \
    --output-dir benchmarks/results
  ```

- GPU adapter smoke (BioBERT):
  ```bash
  python benchmarks/benchmark_medical.py \
    --model biobert --batch-sizes 1 --seq-lengths 64 \
    --num-iterations 3 --warmup-iterations 1 \
    --precision fp16 --device cuda \
    --output-dir benchmarks/results
  ```

- CPU training smoke (synthetic tiny):
  ```bash
  python benchmarks/benchmark_training.py \
    --epochs 1 --batch-size 8 --seq-length 64 --dataset-size 128 \
    --device cpu \
    --output benchmarks/results/train_smoke_cpu.json
  ```

- GPU training smoke (synthetic tiny):
  ```bash
  python benchmarks/benchmark_training.py \
    --epochs 1 --batch-size 8 --seq-length 64 --dataset-size 128 \
    --device cuda \
    --output benchmarks/results/train_smoke_gpu.json
  ```

- Real-adapter training smoke (CPU, BioBERT):
  ```bash
  python benchmarks/benchmark_training.py \
    --use-real-adapter --adapter biobert \
    --epochs 1 --batch-size 4 --dataset-size 64 \
    --device cpu \
    --output benchmarks/results/train_real_adapter_biobert_cpu.json
  ```

- Real-adapter training smoke (CPU, ClinicalBERT):
  ```bash
  python benchmarks/benchmark_training.py \
    --use-real-adapter --adapter clinicalbert \
    --epochs 1 --batch-size 4 --dataset-size 64 \
    --device cpu \
    --output benchmarks/results/train_real_adapter_clinicalbert_cpu.json
  ```

## Optional dataset-based checks

`benchmark_medical.py` can run a tiny classification baseline using a public HF dataset slice:
```bash
python benchmarks/benchmark_medical.py \
  --model biobert --batch-sizes 1 --seq-lengths 64 \
  --num-iterations 1 --warmup-iterations 0 \
  --precision fp32 --device cpu \
  --output-dir benchmarks/results \
  --hf-dataset pubmed_qa --hf-subset pqa_labeled --hf-split train[:50] --hf-label-column final_decision
```
This gracefully skips if the `datasets` package is unavailable or the dataset cannot be loaded.

## Report generation

Aggregate JSON results into a Markdown report:
```bash
python benchmarks/generate_report.py \
  --results-dir benchmarks/results \
  --output benchmark_report.md
```

## Notes

- Memory usage is collected via `tests.medical.memory_profiler.MemoryProfiler` and included in outputs when available.
- Adapter benchmarks also support optional NER metrics via `--test-ner` against a small JSONL fixture.
