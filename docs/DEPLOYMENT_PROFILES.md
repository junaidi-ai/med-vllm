# Deployment Profiles

Simple example profiles describing typical Med vLLM deployments.

Profiles live under `configs/deployment/` and can be used to inform engine settings.

## CPU (dynamic int8)

File: `configs/deployment/cpu.json`

- Device: CPU
- Quantization: PyTorch dynamic int8 (Linear/LayerNorm)
- Intended for: low-memory servers or edge devices without GPU

```json
{
  "profile": "cpu",
  "engine": {
    "device": "cpu",
    "tensor_parallel_size": 1,
    "quantization_bits": 8,
    "quantization_method": "dynamic"
  }
}
```

## Schema

Profiles are simple JSON files with this shape. See `docs/deployment_profile.schema.json` for a formal JSON Schema.

- Required:
  - `engine`: object. Keys are merged into `medvllm.config.Config` fields.
- Optional:
  - `profile`: string identifier.
  - `category`: one of `edge`, `onprem`, `cloud`.
  - `description`: string.

Notes:
- Unknown keys inside `engine` (e.g., `device`) are allowed but ignored by the loader; they are kept for forward compatibility.
- Type and semantic validation (e.g., `quantization_bits` vs `quantization_method`) is enforced by `Config` at initialization.

### Mapping to optimization flags

- Edge (CPU): prefer `quantization_method=dynamic` with `quantization_bits=8`, enable memory pooling.
- On-Prem (GPU): enable mixed precision (`enable_mixed_precision=true`, `mixed_precision_dtype=bf16`), prefer Flash Attention (`attention_impl=flash`, `enable_flash_attention=true`), allow TF32 when safe.
- Cloud (Multi-GPU): set `tensor_parallel_size` appropriately, use FP16 or BF16, enable CUDA memory pooling.

## Single GPU (8-bit bnb)

File: `configs/deployment/gpu_8bit.json`

- Device: CUDA
- Quantization: bitsandbytes 8-bit
- Intended for: single-GPU inference with reduced memory footprint

```json
{
  "profile": "gpu_8bit",
  "engine": {
    "device": "cuda",
    "tensor_parallel_size": 1,
    "quantization_bits": 8,
    "quantization_method": "bnb-8bit"
  }
}
```

## Usage

Profiles live under `configs/deployment/` and can be selected by name or by path.

### CLI (recommended)

Apply a profile with the new `--deployment-profile` flag. Any explicit CLI flags override profile values.

```bash
# Use a named profile from configs/deployment/
python -m medvllm.cli inference generate \
  --model <hf_or_path> \
  --text "Explain hypertension" \
  --deployment-profile cpu

# Or point to a custom JSON path
python -m medvllm.cli inference generate \
  --model <hf_or_path> \
  --text "Explain hypertension" \
  --deployment-profile /abs/path/to/custom_profile.json

# Override some options explicitly (takes precedence over profile)
python -m medvllm.cli inference generate \
  --model <hf_or_path> \
  --text "Explain hypertension" \
  --deployment-profile gpu_8bit \
  --quantization-bits 8 \
  --quantization-method bnb-8bit
```

You can also use an environment variable to set a default profile:

```bash
export MEDVLLM_DEPLOYMENT_PROFILE=gpu_8bit
python -m medvllm.cli inference generate --model <hf_or_path> --text "..."
```

Precedence:
- Profile (from flag or `MEDVLLM_DEPLOYMENT_PROFILE`) provides defaults
- Explicit CLI flags override the profile

For GPU 8-bit, ensure CUDA + bitsandbytes are available.

### Auto-selection (no profile specified)

When you do not pass `--deployment-profile` and `MEDVLLM_DEPLOYMENT_PROFILE` is unset, Med-vLLM will best-effort auto-select a profile based on detected hardware:

- No CUDA available: prefer `edge_cpu_int8.json`, else `cpu.json`.
- CUDA with >=4 GPUs: `cloud_gpu_tp4_fp16.json`.
- CUDA Ampere+ (SM 8.x+) single GPU: `onprem_gpu_bf16_flash.json`.
- Low-memory (<10GB) GPU: `gpu_4bit.json`.
- Otherwise: `gpu_8bit.json`.

This selection is heuristic and may evolve. You can always override by explicitly setting a profile.

## Additional Profiles

The following additional profiles are provided as starting points. Tune as needed for your hardware.

### Edge (CPU, int8)

File: `configs/deployment/edge_cpu_int8.json`

- CPU-only, dynamic int8 quantization
- Emphasizes memory efficiency and pooling for constrained devices

```json
{
  "profile": "edge_cpu_int8",
  "engine": {
    "tensor_parallel_size": 1,
    "quantization_bits": 8,
    "quantization_method": "dynamic",
    "memory_efficient": true,
    "enable_memory_pooling": true,
    "pool_device": "auto"
  }
}
```

### On-Prem (GPU, BF16 + Flash Attention)

File: `configs/deployment/onprem_gpu_bf16_flash.json`

- Single-GPU BF16 mixed precision
- Prefers Flash Attention when available, enables TF32 and cuDNN benchmark

```json
{
  "profile": "onprem_gpu_bf16_flash",
  "engine": {
    "tensor_parallel_size": 1,
    "enable_mixed_precision": true,
    "mixed_precision_dtype": "bf16",
    "enable_flash_attention": true,
    "attention_impl": "flash",
    "allow_tf32": true,
    "cudnn_benchmark": true
  }
}
```

### Cloud (Multi-GPU TP=4, FP16)

File: `configs/deployment/cloud_gpu_tp4_fp16.json`

- Multi-GPU with tensor parallel size 4
- FP16 mixed precision and memory pooling on CUDA

```json
{
  "profile": "cloud_gpu_tp4_fp16",
  "engine": {
    "tensor_parallel_size": 4,
    "enable_mixed_precision": true,
    "mixed_precision_dtype": "fp16",
    "enable_memory_pooling": true,
    "pool_device": "cuda"
  }
}
```

## Clinical Deployment Recommendations

- Edge kiosks/tablets: use `edge_cpu_int8`. Prioritize latency predictability and small memory footprint. Preload models and enable memory pooling.
- On-prem inference servers (Ampere+): use `onprem_gpu_bf16_flash`. Ensure Flash Attention is installed for best performance; enable TF32 when acceptable.
- Cloud autoscaling: use `cloud_gpu_tp4_fp16` (adjust TP to hardware). Combine with batch scheduling and memory pooling. For cost-sensitive scenarios on smaller GPUs, consider `gpu_8bit` or `gpu_4bit`.

Always validate downstream output quality when changing quantization or precision.
