# Quantization in Med vLLM

This document explains how to enable and use 8‑bit and 4‑bit quantization in Med vLLM.

## Quick start (CLI)

- 8‑bit dynamic (CPU):

```bash
python -m medvllm.cli inference generate \
  --model <hf_or_path> \
  --text "Explain hypertension" \
  --quantization-bits 8 \
  --quantization-method dynamic
```

- 8‑bit bitsandbytes (GPU):

```bash
python -m medvllm.cli inference generate \
  --model <hf_or_path> \
  --text "Explain hypertension" \
  --quantization-bits 8 \
  --quantization-method bnb-8bit
```

- 4‑bit NF4 bitsandbytes (GPU):

```bash
python -m medvllm.cli inference generate \
  --model <hf_or_path> \
  --text "Explain hypertension" \
  --quantization-bits 4 \
  --quantization-method bnb-nf4
```

## Programmatic API

- Dynamic int8 (CPU) via `medvllm.optim.quantization.quantize_model`.
- Bitsandbytes load helpers:

```python
from medvllm.optim.quantization import bnb_load_quantized
model = bnb_load_quantized("<hf_or_path>", bits=4, method="bnb-nf4", device_map="auto")
```

- Optimizer utility:

```python
from medvllm.optim.medical_optimizer import MedicalModelOptimizer, OptimizerConfig
import torch

cfg = OptimizerConfig(model_name_or_path="<hf_or_path>")
opt = MedicalModelOptimizer(model, cfg)
model_q = opt.quantize(bits=8, method="dynamic")  # or "bnb-8bit" / "bnb-nf4"
opt.optimize_memory()
res = opt.benchmark(["Patient presents with chest pain."], iterations=10)
print(res)
```

## Notes & requirements

- Bitsandbytes paths require: CUDA + `bitsandbytes` + compatible `transformers`.
- `device_map='auto'` is used for GPU sharding where appropriate.
- Dynamic int8 runs on CPU and uses PyTorch `quantize_dynamic` on Linear/LayerNorm.
- Offline saving of true 4/8‑bit bnb weights is not standardized; see `bnb_offline_hint()`.

## Internals

- Config fields: `quantization_bits`, `quantization_method` in `medvllm/config.py`.
- Model load integration: `medvllm/engine/model_runner/model.py` applies bnb or dynamic quantization.
- Utilities:
  - `medvllm/optim/quantization.py`: dynamic quantization + bnb helpers.
  - `medvllm/optim/medical_optimizer.py`: convenience class for quantize/benchmark/export.

## Benchmarking

Run a quick benchmark:

```bash
python scripts/benchmark_quantization.py --model <hf_or_path> --iters 10
```

This will run baseline, dynamic int8, and bnb‑8bit (if CUDA).

### Accuracy scaffold (tiny dataset)

You can estimate accuracy deltas with a small fixture or your CSV using the scaffold:

```bash
python benchmarks/benchmark_quantization_accuracy.py \
  --model <hf_seq_cls_model> \
  --dataset-csv tests/fixtures/data/datasets/text_classification_dataset.csv \
  --limit 64
```

This compares baseline vs dynamic int8 (CPU) and bnb‑8bit (if CUDA). It is best‑effort and will skip gracefully if dependencies or data are missing.

## Deployment presets

Example deployment profiles are provided under `configs/deployment/`:

- `cpu.json`: CPU inference with dynamic int8
- `gpu_8bit.json`: Single‑GPU inference using bitsandbytes 8‑bit
- `gpu_4bit.json`: Single‑GPU inference using bitsandbytes 4‑bit NF4
