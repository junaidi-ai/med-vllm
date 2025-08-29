# Med-vLLM Benchmarks

This guide explains how to run the medical adapter benchmarks and interpret results.

## Requirements
- Python, PyTorch installed (GPU optional)
- transformers for model downloads
- Optional CUDA GPU with recent NVIDIA driver
- Internet access for first-time model downloads (cached afterward)

## Quick CPU Smoke Test
Runs a minimal configuration to validate the pipeline and produce JSON results.

```bash
python3 benchmarks/benchmark_medical.py \
  --model biobert \
  --device cpu \
  --num-iterations 2 \
  --warmup-iterations 1 \
  --batch-sizes 1 \
  --seq-lengths 64 \
  --output-dir benchmarks/results
```

Output JSON files are written to the chosen `--output-dir` (default now `benchmarks/results/`). Each file includes:
- Performance: `avg_latency_ms`, `tokens_per_second`
- Memory: `memory_usage_mb` (CPU always; GPU when `--device cuda`)
- Run metadata: `model_type`, `batch_size`, `seq_length`, `precision`, `device`

To debug file writes, you may add `--debug-io` to print exact JSON paths and sizes.

## GPU Run (optional)
Requires an up-to-date NVIDIA driver. See `docs/GPU.md` for the full GPU benchmarking checklist. Example:

```bash
python3 benchmarks/benchmark_medical.py \
  --model clinicalbert \
  --device cuda \
  --precision fp16 \
  --num-iterations 10 \
  --warmup-iterations 2 \
  --batch-sizes 1 4 8 \
  --seq-lengths 128 256 512 \
  --output-dir benchmarks/results
```

If CUDA is requested but not available, the script automatically falls back to CPU and prints an informational message. To silence it, set `--device cpu` explicitly.

## Notes
- Models are downloaded from Hugging Face on first run and cached locally.
- Memory profiling uses `tests/medical/memory_profiler.py` and reports CPU keys always and GPU keys when running with CUDA.
- To control cache location:
  - `export HF_HOME=~/.cache/huggingface`
- For offline runs (after models are cached):
  - `export TRANSFORMERS_OFFLINE=1`

## Classification Testing (optional)
See `docs/classification_tests.md` for majority-class baseline evaluation on a small fixture dataset. Enable via `--test-accuracy` and optionally `--dataset-csv`.

## Troubleshooting
- CUDA driver warnings: update the system NVIDIA driver or run on CPU.
- No GPU memory keys in results: expected when `--device cpu` or CUDA unavailable.
