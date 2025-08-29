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

Use the profile values when constructing `LLMEngine`/`Config` or the CLI flags:

```bash
python -m medvllm.cli inference generate \
  --model <hf_or_path> \
  --text "Explain hypertension" \
  --quantization-bits 8 \
  --quantization-method dynamic
```

For GPU 8-bit, switch method to `bnb-8bit` and ensure CUDA + bitsandbytes are available.
