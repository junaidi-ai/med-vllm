# GPU Benchmarking Checklist

This guide walks you through preparing your system to run GPU benchmarks with `benchmarks/benchmark_medical.py`, verifying CUDA availability, and running a GPU smoke test.

## Prerequisites
- An NVIDIA GPU and a recent NVIDIA driver
- Python virtualenv for this repo (e.g., `./.venv`)
- Project dependencies installed

## 1) Verify NVIDIA driver and GPU visibility
- Check driver/GPU status:
```bash
nvidia-smi
```
- If the command is not found or shows no GPU, install the appropriate NVIDIA driver for your system and reboot. See: https://www.nvidia.com/Download/index.aspx

## 2) Check PyTorch CUDA setup
- Confirm whether your PyTorch build has CUDA, and whether CUDA is available at runtime:
```bash
/home/kd/med-vllm/.venv/bin/python - <<'PY'
import torch
print('torch_version =', torch.__version__)
print('torch_cuda_build =', getattr(torch.version, 'cuda', None))
print('cuda_available =', torch.cuda.is_available())
print('device_count =', torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print('device_name =', torch.cuda.get_device_name(0))
PY
```
- Goal: `cuda_available = True` and a non-zero `device_count`.

## 3) Align driver and PyTorch CUDA versions
- If your driver is too old for the installed PyTorch CUDA build (you may see warnings like “driver too old”), you have two options:
  - Update the NVIDIA driver to a version compatible with your PyTorch CUDA build
  - OR install a PyTorch wheel matching your driver’s supported CUDA version
- Example (adjust per https://pytorch.org/get-started/locally/):
```bash
# Example: install PyTorch with CUDA 12.1 wheels
/home/kd/med-vllm/.venv/bin/pip install --upgrade \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```
- If you cannot use GPU, you can still benchmark on CPU with `--device cpu`.

## 4) Optional performance extras
- Flash Attention can improve performance on supported GPUs:
```bash
/home/kd/med-vllm/.venv/bin/pip install flash-attn --no-build-isolation
```
- If not installed, the code will warn and fall back to a standard attention implementation.

## 5) Run a GPU smoke test
- Once CUDA is available, run a minimal GPU benchmark:
```bash
/.venv/bin/python benchmarks/benchmark_medical.py \
  --model biobert \
  --device cuda \
  --precision fp16 \
  --num-iterations 2 \
  --warmup-iterations 1 \
  --batch-sizes 1 \
  --seq-lengths 16 \
  --output-dir benchmarks/results \
  --debug-io
```
- Expect JSON files in `benchmarks/results/` and GPU memory keys in results.
- If CUDA is requested but unavailable, the script will automatically fall back to CPU and print an informational message. To avoid the message, set `--device cpu` explicitly.

## 6) Scale up benchmarks
- Increase `--batch-sizes` (e.g., `1 4 8`) and `--seq-lengths` (e.g., `128 256 512`).
- Keep `--precision fp16` for better throughput, or compare against `fp32`.
- Optionally toggle KV cache with `--no-kv-cache` to compare effects on throughput/latency.

## Useful flags and tips
- Debug JSON writes: add `--debug-io`.
- Classification metrics on fixtures: add `--test-accuracy` and optionally `--dataset-csv`.
- Control HF cache: `export HF_HOME=~/.cache/huggingface`
- Offline mode once models cached: `export TRANSFORMERS_OFFLINE=1`

## Troubleshooting
- "NVIDIA driver too old" warning from PyTorch:
  - Update your system’s NVIDIA driver or install a compatible PyTorch CUDA wheel.
- `nvidia-smi` missing or shows no devices:
  - Install NVIDIA drivers for your OS and reboot; ensure GPU is accessible (container/VM passthrough if applicable).
- `cuda_available = False` despite a driver:
  - Driver/runtime mismatch. Reinstall PyTorch with a CUDA build matching your driver, or update the driver.
- No GPU memory keys in JSON results:
  - Expected when running on CPU or when CUDA is not available.
