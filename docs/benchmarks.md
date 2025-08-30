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

## Imaging Convolution Benchmark

File: `benchmarks/benchmark_imaging.py`

- Purpose: quick performance and memory benchmark of a tiny CNN (conv/pool stacked with adaptive pool + FC), useful for testing AMP, channels-last, cuDNN autotuning, and optional `torch.compile` fusion.
- Device: auto-detects CUDA; use `--device {cpu|cuda}` to force.
- Precision: `--dtype {fp16|bf16|fp32}` with optional `--no-amp` to disable autocast on CUDA.
- Memory format: `--channels-last` to use NHWC on CUDA.
- cuDNN autotune: `--cudnn-benchmark` to enable autotuner.
- Compile: `--compile` to enable `torch.compile` (Inductor) when available.
- Accuracy check: `--acc-check` runs two forward passes on the same input and reports numerical stats.

Examples:

```bash
# CPU smoke
python benchmarks/benchmark_imaging.py \
  --device cpu --batches 2 --batch 1 --width 64 --height 64 \
  --no-amp --acc-check --out benchmark_results_cpu_smoke/conv_smoke_acc.json

# CUDA with AMP, channels-last, cuDNN autotuner, torch.compile
python benchmarks/benchmark_imaging.py \
  --device cuda --dtype fp16 --batch 16 --batches 50 \
  --channels-last --cudnn-benchmark --compile --acc-check \
  --out benchmarks/results/conv_cuda.json
```

Output JSON (subset of keys):

```json
{
  "device": "cuda",
  "dtype": "fp16",
  "channels_last": true,
  "amp": true,
  "cudnn_benchmark": true,
  "compiled": true,
  "input_shape": [16, 1, 256, 256],
  "batches": 50,
  "batch_time_ms": 3.21,
  "imgs_per_sec": 4985.4,
  "cpu_max_rss_mb": 512.3,
  "cuda_max_mem_mb": 824.6,
  "acc_check_enabled": true,
  "has_nan": false,
  "has_inf": false,
  "repeatability_pass": true,
  "max_abs_diff": 0.012,
  "mean_abs_diff": 0.0001
}
```

Notes:
- When AMP is enabled on CUDA, small differences between repeated runs are expected; thresholding is handled internally for the pass/fail flag.
- CPU runs report only CPU memory keys; `cuda_max_mem_mb` is null.

## Classification Testing (optional)
See `docs/classification_tests.md` for majority-class baseline evaluation on a small fixture dataset. Enable via `--test-accuracy` and optionally `--dataset-csv`.

## Troubleshooting
- CUDA driver warnings: update the system NVIDIA driver or run on CPU.
- No GPU memory keys in results: expected when `--device cpu` or CUDA unavailable.
