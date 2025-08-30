# Optimization and Profiling Guide

This document describes how to profile, benchmark, and optimize medical NLP and imaging workloads in this repository. It covers attention backends and fallbacks, unified profiling, imaging convolution variants and flags, and our operation fusion plan/decision.

## Attention Backends and Fallbacks

- backends: manual, PyTorch SDPA, Flash Attention (if installed)
- selection: runtime, with graceful fallback and warnings in `medvllm/layers/attention.py` and `benchmarks/benchmark_attention.py`.
- KV cache optimization: Triton-powered efficient storage is integrated in `medvllm/layers/attention.py`.

Recommended:
- Install flash-attn on compatible GPUs for best performance.
- If unavailable, SDPA provides good defaults; otherwise manual attention is used.

## Unified Profiling

A single API wraps `torch.profiler` (pref) with fallback to a lightweight memory profiler.

- Entry point: `medvllm.utils.profiler.get_profiler(device, emit_trace=False, trace_dir=None)`
- Usage pattern:
  ```python
  profiler = get_profiler(device="cuda", emit_trace=True, trace_dir="./profiles")
  with profiler.profile():
      # run workload
  results = profiler.results  # cpu/gpu time, memory, optional trace path
  ```

- Benchmarks integrated:
  - `benchmarks/benchmark_attention.py` (flags: `--emit-trace`, `--trace-dir`)
  - `benchmarks/benchmark_imaging.py` (flags: `--emit-trace`, `--trace-dir`)

## Imaging Convolution Variants and Flags

`benchmarks/benchmark_imaging.py` supports 2D/3D, grouped/depthwise convs, and common perf flags.

- Dimensionality and grouping:
  - `--conv-type {2d,3d}`: choose 2D or 3D convs
  - `--in-ch N`: input channels
  - `--groups G`: grouping for all conv layers
  - `--depthwise`: sets `groups=in_ch` for depthwise (overrides `--groups`)
  - `--depth D`: depth for 3D inputs

- Performance flags:
  - `--channels-last`: use channels_last (2D) or channels_last_3d (3D) formats
  - `--cudnn-benchmark`: enable cudnn autotuner (GPU only)
  - `--no-amp`: disable AMP (default uses AMP bf16/fp16 on CUDA)
  - `--compile`: try `torch.compile(mode="max-autotune")`
  - `--emit-trace`, `--trace-dir`: enable torch.profiler traces

- Outputs: JSON with throughput, per-batch latency, and CPU/GPU memory maxima. Optional repeatability metrics with `--acc-check`.

Examples:
- Depthwise 2D on CUDA bf16 with channels_last and compile:
  ```bash
  python benchmarks/benchmark_imaging.py --conv-type 2d --in-ch 32 --depthwise \
    --device cuda --dtype bf16 --channels-last --compile --emit-trace
  ```
- 3D grouped convs on CPU fp32:
  ```bash
  python benchmarks/benchmark_imaging.py --conv-type 3d --in-ch 4 --groups 2 \
    --device cpu --dtype fp32 --height 64 --width 64 --depth 16
  ```

## Medical Accuracy Checks

`benchmarks/benchmark_medical.py` provides:
- Adapter performance benchmarking (latency, throughput, memory)
- NER metrics on JSONL fixtures via `medvllm.utils.ner_metrics`
  - Defaults to tests fixture; falls back to `benchmarks/datasets/i2b2_ner_sample.jsonl`
- Classification metrics:
  - CSV fixture baseline (majority) with accuracy/precision/recall/F1
  - HF dataset small-slice majority baseline (optional)
  - NEW: JSONL-based classification on included samples:
    - `benchmarks/datasets/pubmed_sample.jsonl`
    - `benchmarks/datasets/mimic_notes_sample.jsonl`
  - AUC (macro/micro OVR) computed when scores are present; for the majority baseline we provide degenerate one-hot scores to allow indicative AUC values.

## Operation Fusion Plan / Decision

- Current state:
  - Attention: Flash Attention (if available) + Triton KV-cache; robust fallbacks.
  - FFN: Efficient PyTorch modules; no custom Triton fused kernels for GEMM+activation+dropout yet.
  - Depthwise Conv2D: Triton CUDA kernel (optional) with PyTorch eager fallback.

- Decision:
  - Prefer `torch.compile` (Inductor) first for op fusion and kernel selection across platforms, minimizing maintenance burden.
  - Where `torch.compile` underperforms in hotspots (e.g., softmax*V in long-seq attention, fused GLU, depthwise conv blocks), implement targeted Triton kernels with clear benchmarking guardrails.

- Next steps:
  - Track top kernels via unified profiler traces.
  - If speedups >10-15% are consistently achievable with custom Triton on target hardware, implement optional fused kernels behind flags, with correctness/unit tests.

### Softmax×V (Triton placeholder)

- File: `medvllm/kernels/triton_softmaxv.py`
- Status: placeholder fused path; currently slower than eager/SDPA on common CUDA setups.
- Enable flag:
  - `MEDVLLM_ENABLE_TRITON_SOFTMAXV=1` — opt-in to the Triton path. Default is disabled.
- Recommendation: keep disabled unless benchmarking/experimenting. Prefer FlashAttention or SDPA.

Benchmarking and CI:
- Use `benchmarks/benchmark_attention.py` to compare eager vs Triton softmax×V. New flag `--enable-triton-softmaxv` sets the env var for this run.
  ```bash
  python benchmarks/benchmark_attention.py \
    --seq 512 --heads 8 --dim 64 --iters 50 \
    --device cuda --dtype bf16 \
    --attn-softmaxv-bench --enable-triton-softmaxv
  ```
- Nightly CUDA CI runs a non-blocking softmax×V microbench and uploads JSON to artifacts: see `.github/workflows/cuda-ci.yml` (step: "Softmax×V CUDA microbench (non-blocking)").

### Depthwise/Separable Conv3D (Triton, opt-in)

- File: `medvllm/kernels/triton_separable_conv3d.py`
- Status: optional Triton CUDA kernel for depthwise 3D (3x3x3), stability-first, improving performance via guarded vectorization and minimal autotune set.
- Enable flag:
  - `MEDVLLM_ENABLE_TRITON_SEP3D=1` — opt-in to the Triton depthwise 3D path. Default disabled.
- Tuning env overrides (for sweeps/CI):
  - `MEDVLLM_SEP3D_BLOCK_W` in {64,128,256}
  - `MEDVLLM_SEP3D_WARPS` in {2,4,8}
  - `MEDVLLM_SEP3D_STAGES` in {2,3,4}
- Fused options (prototype):
  - `MEDVLLM_SEP3D_FUSE_BIAS=1` — fuse bias add in-kernel
  - `MEDVLLM_SEP3D_FUSE_RELU=1` — fuse ReLU activation in-kernel

Microbenchmarking and CI:
- Run dedicated 3D depthwise microbench:
  ```bash
  MEDVLLM_ENABLE_TRITON_SEP3D=1 \
  python benchmarks/benchmark_separable_conv3d.py
  ```
- Nightly CUDA CI runs a non-blocking separable 3D microbench: see `.github/workflows/cuda-ci.yml` (step: "Separable 3D CUDA microbench (non-blocking)").

Accuracy preservation:
- New tests under `tests/accuracy/test_kernels_accuracy.py` compare eager vs Triton/fused outputs across dtypes with tolerances:
  - fp32: `atol=1e-5, rtol=1e-4`
  - bf16/fp16: `atol=5e-3, rtol=5e-3`
  These tests fail if deltas exceed thresholds.

## Depthwise Conv2D (Triton)

- File: `medvllm/kernels/triton_depthwise_conv2d.py`
- API:
  - `build_fused_depthwise_conv2d_if_available(C, K, stride=1, padding=1, dilation=1)` → `nn.Module | None`
  - Fallback: PyTorch grouped conv (`groups=C`) when Triton/CUDA unavailable
- Environment toggles:
  - `MEDVLLM_DISABLE_TRITON=1` — force-disable Triton path
  - Autotuning: kernel uses Triton autotune over a small set of tile sizes to pick `(BLOCK_H, BLOCK_W, num_warps)` per shape at runtime (no behavior change; improves perf).

Compatibility:
- Kernel supports arbitrary K (odd/even), stride>=1, dilation>=1; use appropriate padding for your configuration.
- Inputs in channels-last (NHWC) are accepted; they are converted to contiguous NCHW for the kernel and outputs are restored to channels-last for convenience. A true NHWC-optimized kernel is planned.

Runtime requirements:
- CUDA device (NVIDIA) and `triton` package installed.
- Kernel tiles the output spatial grid and iterates K×K per channel.

### Microbenchmarking

Use `benchmarks/benchmark_imaging.py` depthwise bench:

```bash
# Single case (uses current size flags)
python benchmarks/benchmark_imaging.py \
  --conv-type 2d --in-ch 32 --height 256 --width 256 \
  --device cuda --dtype fp32 --batch 8 \
  --depthwise-bench --depthwise-bench-iters 100 \
  --emit-trace --trace-dir ./profiles

# Multiple cases
python benchmarks/benchmark_imaging.py \
  --device cuda --conv-type 2d --batch 8 \
  --depthwise-bench --depthwise-bench-iters 50 \
  --depthwise-bench-sizes 8x128x128,32x256x256,64x512x512
```

Output JSON section (`depthwise_bench`) contains per-case eager vs fused timings. When CUDA/Triton are unavailable, `fused.available` is `false`.

Tip: add `--channels-last` to compare NHWC vs NCHW, and `--compile` to see interaction with Inductor.

## Known Fallbacks and Tips

- If `torch.profiler` is unavailable, the profiler falls back to memory-only stats.
- Channels-last/3D memory formats are CUDA-optimized; on CPU, gains may vary.
- `cudnn.benchmark` improves perf for static shapes; disable for dynamic shapes.
- AMP bf16 is recommended on recent NVIDIA GPUs; fp16 may be faster on some devices but check numerics.

## Reproducibility

- Save all benchmark JSONs under versioned directories.
- Include `--emit-trace` and archive trace files for deeper analysis.
- Record commit SHA and environment in reports where possible.
