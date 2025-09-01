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

## Hardware Matrix (multi-backend)

File: `benchmarks/benchmark_hardware_matrix.py`

- Purpose: run small, comparable forward passes across backends and devices and write a per-run JSON plus a consolidated CSV and `summary.md`.
- Backends: `--backend {auto|torch|onnxruntime|openvino|tensorrt|tpu}`
  - Implemented minimal paths: `torch` (covers `cpu`, `cuda`, `mps`, `xpu`), `onnxruntime`, `openvino`, `tensorrt` (via ONNX Runtime TensorRT EP), and `tpu` (torch_xla).
- Edge INT8: pass `--int8` to enable dynamic INT8 quantization on CPU. If present, `configs/deployment/edge_cpu_int8.json` is read for advisory knobs.
- Accuracy parity: pass `--acc-check` to compute accuracy parity vs a torch CPU fp32 reference on shared inputs. Metrics recorded per backend: `acc_mse`, `acc_cosine`, `acc_top1_match`, `acc_kl` and `acc_error` when checks cannot be computed.

Examples:

```bash
# CPU torch baseline
python3 benchmarks/benchmark_hardware_matrix.py \
  --models biobert clinicalbert \
  --backend torch --batch-size 4 --seq-length 256

# ONNX Runtime on CPU (uses onnx export internally)
python3 benchmarks/benchmark_hardware_matrix.py \
  --models biobert --backend onnxruntime --batch-size 4 --seq-length 256

# OpenVINO CPU path
python3 benchmarks/benchmark_hardware_matrix.py \
  --models clinicalbert --backend openvino --batch-size 4 --seq-length 256

# TensorRT (via ONNX Runtime TensorRT EP; requires TensorRT + ORT build with TRT EP)
python3 benchmarks/benchmark_hardware_matrix.py \
  --models biobert --backend tensorrt --batch-size 4 --seq-length 256

# INT8 dynamic quant on CPU
python3 benchmarks/benchmark_hardware_matrix.py \
  --models biobert --backend torch --int8 --batch-size 4 --seq-length 256

# TPU (torch_xla) with conservative defaults applied internally
python3 benchmarks/benchmark_hardware_matrix.py \
  --models biobert --backend tpu

# With accuracy parity checks (adds metrics to summaries)
python3 benchmarks/benchmark_hardware_matrix.py \
  --models biobert clinicalbert --backend auto --acc-check --batch-size 4 --seq-length 128
```

Outputs:

- Directory: `reports/<YYYY-MM-DD>/hardware_matrix/`
- Files per run: `*.json` entries with latency, tokens/sec, memory from `tests/medical/memory_profiler.py`, and accuracy metrics when `--acc-check` is used.
- Consolidated: `summary.csv`, `summary.json`, and `summary.md` (table for quick viewing) include accuracy columns: `acc_mse`, `acc_cosine`, `acc_top1_match`, `acc_kl` (and `acc_error` if applicable) plus a `status` column indicating `ok` or `acc_error`.

Notes:

- If a backend package is unavailable (e.g., OpenVINO not installed), the script records a `load_error` row instead of crashing.
- `--precision fp16` is coerced to `fp32` on devices that do not support half precision.
 - TensorRT path uses ONNX export and ONNX Runtime TensorRT Execution Provider. Ensure your environment has a TRT-enabled ORT build and NVIDIA drivers.
 - Torch backend automatically targets available devices: CPU, CUDA, Apple MPS, or Intel XPU when present.

### Native TensorRT path

When `--backend tensorrt` is selected, the benchmark attempts a native TensorRT build and run first, and falls back to ONNX Runtime's TensorRT EP if unavailable.

- Engine caching: serialized engines are cached under `reports/<date>/hardware_matrix/trt_cache/` keyed by ONNX hash, TRT version, precision flags, and optimization profile.
  - Default: cache enabled. Disable with `--no-trt-cache`.
- Precision flags:
  - `--trt-bf16`: sets BF16 flag when supported by your TRT build and GPU.
  - `--trt-fp8`: sets FP8 flag when supported (newer H100-class).
- Optimization profile ranges for dynamic shapes:
  - `--trt-min-batch/--trt-opt-batch/--trt-max-batch`
  - `--trt-min-seq/--trt-opt-seq/--trt-max-seq`
  - If omitted, the profile is fixed to the requested `--batch-size` and `--seq-length`.

Examples:

```bash
# Build and cache a TRT engine for a range of batch/seq sizes (with BF16 if supported)
python3 benchmarks/benchmark_hardware_matrix.py \
  --models biobert --backend tensorrt --batch-size 8 --seq-length 256 \
  --trt-bf16 \
  --trt-min-batch 1 --trt-opt-batch 8 --trt-max-batch 16 \
  --trt-min-seq 64 --trt-opt-seq 256 --trt-max-seq 512

# Disable caching and attempt FP8 build (requires compatible TRT/GPU)
python3 benchmarks/benchmark_hardware_matrix.py \
  --models clinicalbert --backend tensorrt --batch-size 4 --seq-length 128 \
  --no-trt-cache --trt-fp8
```

## Remaining items on TODO

- **TPU validation**: Must run on a real TPU VM.
- **Specialized medical hardware**: Needs the actual device (cannot be simulated meaningfully).
