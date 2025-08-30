# Med vLLM CLI

This document describes the command-line interface exposed by `medvllm.cli`.

## I/O patterns

- **--text**: Provide inline input (prompt or text) directly on the command line.
- **--input FILE**: Read input contents from a file on disk.
- **stdin (pipe)**: If neither `--text` nor `--input` is given and stdin is piped, the CLI reads from stdin.
  - Example: `cat note.txt | python -m medvllm.cli inference ner --json-out`
- **--output FILE**: Write command results to a file.
  - NER: writes JSON when `--json-out` is set; otherwise writes the rendered table text.
  - Generate: writes generated text to `FILE`. If `--json-meta` is set, also writes `FILE.meta.json`.
  - Classification: always writes JSON to `FILE` when provided.

- Entry point module: `medvllm/cli/__init__.py`
- Command groups registered:
  - `model` — model registry management
  - `inference` — run inference tasks (NER, generation, classification)
  - `training` — experimental training utilities

## Command Tree (Diagram)

```

JSON output schema (consistent across modes):
```json
{
  "entities": [
    { "text": "HTN", "type": "condition", "start": 10, "end": 13, "confidence": 0.92,
      "ontology_links": [ { "ontology": "UMLS", "code": "C0020538", "name": "Hypertension", "score": 0.91 } ]
    }
  ]
}
```
- `confidence` may be absent/0.0 in processor-only mode.
- `ontology_links` present unless `--no-link`.
medvllm
└── (root CLI)
    ├── model
    │   ├── register
    │   ├── unregister
    │   ├── list
    │   ├── info
    │   └── clear-cache
    ├── inference
        ├── ner
        ├── generate
        └── classification
    └── training
        └── train
```

## Getting Help

- Top-level help:
  ```bash
  python -m medvllm.cli --help
  ```
- Help alias:
  - All commands support `-h` as a short alias for `--help`.
  ```bash
  python -m medvllm.cli -h
  python -m medvllm.cli inference ner -h
  ```
- Group help (e.g., inference):
  ```bash
  python -m medvllm.cli inference --help
  ```
- Subcommand help (e.g., NER):
  ```bash
  python -m medvllm.cli inference ner --help
  ```

## Input Methods

Each inference subcommand supports one or more of the following:
- `--text` "inline text"
- `--input path/to/file.txt`
- Stdin (pipe): `cat note.txt | python -m medvllm.cli inference ner`

If no input is provided, the CLI reads from stdin when piped; otherwise it errors.

Notes on input size:
- If `--input` file size exceeds ~2MB, the CLI prints a warning that processing may be slow.
- If stdin input exceeds ~20k characters, the CLI prints a warning suggesting a smaller excerpt.

## Inference Commands

### NER
File: `medvllm/cli/inference_commands.py` → `cmd_ner`

- Description: Medical NER via `NERProcessor` with dual-mode operation:
  - Processor-only (default): fast regex/gazetteer fallback, no external deps.
  - Model-backed (optional): specify `--model` to use a Hugging Face token-classification pipeline via an internal adapter; output normalized to the same schema.

- Options:
  - `--text, --input` — input source
  - `--input-format [auto|text|pdf]` — auto-detects by extension when `auto`
  - `--model NAME` — optional model id or registered name for model-backed NER
  - `--ontology UMLS` — ontology to use for linking (default: `UMLS`)
  - `--no-link` — disable ontology linking
  - `--json-out` — print JSON
  - `--output FILE` — write output to file. Writes JSON when `--json-out` is set; otherwise writes the rendered table.

- Model/task validation:
  - When `--model` is provided, the CLI validates that the model supports the `ner` task using the model registry metadata (`capabilities["tasks"]`).
  - If the model is registered and does not list `ner`, the command fails with: `Model '<name>' does not support task 'ner'`.
  - If the model is not registered, a warning is shown and validation is skipped.
  - See also: `python -m medvllm.cli model list-capabilities`.

Examples:
```bash
# Quick inline
python -m medvllm.cli inference ner --text "Patient with HTN and DM."

# From file with JSON
python -m medvllm.cli inference ner --input note.txt --json-out

# Specify ontology
python -m medvllm.cli inference ner --text "on metformin" --ontology UMLS

# Save to file (JSON)
python -m medvllm.cli inference ner --input note.txt --json-out --output ner.json

# Save to file (table text)
python -m medvllm.cli inference ner --input note.txt --output ner.txt

# Model-backed NER with validation (model must support 'ner')
python -m medvllm.cli inference ner \
  --text "Patient with Hypertension." \
  --model biobert-base-cased-v1.2 \
  --json-out
```

Output (table mode):
```
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Text     ┃ Type      ┃ Span ┃ Link (top)    ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━┩
│ HTN      │ CONDITION │ 10–13│ UMLS:C0020538 │
└──────────┴───────────┴──────┴───────────────┘
```

### Generate
File: `medvllm/cli/inference_commands.py` → `cmd_generate`

- Description: Medical text generation via `TextGenerator` with strategy and constraints.
- Required:
  - `--model` — model name/path to use for generation
- Options:
  - Sampling/strategy: `--strategy [greedy|sampling|beam]`, `--max-length`, `--temperature`, `--top-p`, `--top-k`, `--beam-width`
  - Styling/purpose: `--purpose`, `--readability`, `--tone`, `--structure`, `--specialty`
  - Length targets: `--target-words`, `--target-chars`
  - Misc: `--no-disclaimer` (skip automatic disclaimer), `--json-meta` (print metadata JSON)
  - `--output FILE` — write generated text to FILE. If `--json-meta` is used, metadata is also written to `FILE.meta.json`.
  - Profiling: `--profile`, `--profiler-device {auto|cpu|cuda}`, `--emit-trace`, `--trace-dir DIR`
    - `--emit-trace` enables exporting a Chrome trace via torch.profiler.
    - `--trace-dir` selects the directory for trace files (default: `./profiles/`).
  - Compilation (op fusion): `--compile/--no-compile`, `--compile-mode {default|reduce-overhead|max-autotune}`
    - Enables `torch.compile` to allow inductor-based fusion on supported setups.

#### Attention backends and flags
- Flags:
  - `--attention-impl {auto|flash|sdpa|manual}` — select backend. `auto` (default) prefers Flash → SDPA → manual based on availability.
  - `--flash-attention/--no-flash-attention` — user preference to enable/disable FlashAttention when available.

- Conflicts and warnings:
  - Using `--attention-impl flash` together with `--no-flash-attention` emits a CLI warning and disables Flash at runtime.

- Runtime fallback behavior (in `medvllm/layers/attention.py` → `Attention.forward()`):
  - If `flash` requested but FlashAttention is unavailable (package/CUDA), a warning is emitted, then fallback to SDPA if available, else manual.
  - If `sdpa` requested but PyTorch SDPA is unavailable, a warning is emitted, then fallback to Flash (if available), else manual.
  - With `auto`, the backend is chosen in order: Flash (if available) → SDPA → manual.

- Notes:
  - Fallbacks are best-effort and non-fatal to preserve UX. Warnings explain what happened.
  - The `--flash-attention` flag is a hint; actual selection depends on runtime availability of CUDA and the `flash-attn` package.

#### Memory pooling (experimental)
- Flags (Generate):
  - `--memory-pooling/--no-memory-pooling` — enable/disable tensor memory pooling for intermediate tensors.
  - `--pool-max-bytes N` — optional cap (in bytes) for the memory pool. Default: unlimited.
  - `--pool-device {auto|cpu|cuda}` — device that backs the memory pool. Default: `auto`.

- Behavior:
  - When enabled, the engine attempts to reuse allocated buffers for intermediate tensors, reducing allocator churn and fragmentation.
  - Pool device defaults to `auto` and will pick CUDA when available for GPU models, else CPU.
  - Max bytes is a soft upper bound; exact enforcement is implementation-dependent.

- Conflicts and warnings:
  - Supplying `--pool-*` while using `--no-memory-pooling` prints a CLI warning and ignores pool options.

Examples:
```bash
# Enable pooling with a 2GB cap on CUDA
python -m medvllm.cli inference generate \
  --text "Explain HTN." \
  --model your-hf-model \
  --memory-pooling --pool-max-bytes $((2*1024*1024*1024)) --pool-device cuda

# Disable pooling explicitly (pool options ignored with a warning)
python -m medvllm.cli inference generate \
  --text "Explain HTN." \
  --model your-hf-model \
  --no-memory-pooling --pool-max-bytes 1048576
```

Examples:
```bash
# Patient-friendly summary with beam strategy and metadata JSON
python -m medvllm.cli inference generate \
  --text "Explain hypertension to a patient." \
  --model your-hf-model \
  --strategy beam --beam-width 3 \
  --purpose patient --readability general --tone informal \
  --target-words 120 --json-meta

# With profiling trace exported to ./profiles
python -m medvllm.cli inference generate \
  --text "Explain HTN." \
  --model your-hf-model \
  --profile --emit-trace --trace-dir ./profiles

# Enable inductor compile with max-autotune
python -m medvllm.cli inference generate \
  --text "Explain HTN." \
  --model your-hf-model \
  --compile --compile-mode max-autotune

# Save generated text and metadata
python -m medvllm.cli inference generate \
  --text "Explain HTN to a patient." \
  --model your-hf-model \
  --json-meta \
  --output out.txt
# Produces: out.txt (text) and out.txt.meta.json (metadata)
```

Output (text plus JSON metadata when `--json-meta` is set):
```
[generated text...]
{
  "strategy": "beam",
  "temperature": 0.7,
  "constraints_report": { ... }
}
```

### Classification
File: `medvllm/cli/inference_commands.py` → `cmd_classification`

- Description: Lightweight text classification using Hugging Face `transformers` pipeline.
- Note: Requires `transformers` installed. The CLI errors with a clear message if unavailable.
- Options:
  - `--model` — HF model id (default: `distilbert-base-uncased-finetuned-sst-2-english`)
  - `--json-out` — print JSON
  - `--output FILE` — write JSON result to FILE (always JSON for file output).

Example:
```bash
python -m medvllm.cli inference classification \
  --text "The medication worked well." \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  --json-out

# Save JSON result
python -m medvllm.cli inference classification \
  --text "The medication worked well." \
  --json-out --output cls.json
```

JSON output:
```json
{"label": "POSITIVE", "score": 0.9876}
```

## Model Commands (Summary)
File: `medvllm/cli/model_commands.py`

- Manage model registry (register, unregister, list, info, clear-cache)
- Usage examples:
```bash
python -m medvllm.cli model list
python -m medvllm.cli model info --name biobert-base-cased-v1.2

# List declared task capabilities (used by NER validation)
python -m medvllm.cli model list-capabilities

# JSON output
python -m medvllm.cli model list-capabilities --json
```

## Training (Experimental)
File: `medvllm/cli/training_commands.py`

- Description: Minimal utilities to run training loops using the lightweight `MedicalModelTrainer`.
- Commands:
  - `training train` — run a simple training job.

See also: the full trainer guide in `docs/TRAINER.md` for configuration options, checkpointing, AMP, logging, and export.

Options (subset):
- `--epochs`, `--batch-size`, `--lr`, `--amp/--no-amp`, `--device {auto|cpu|cuda|xla}`, `--output`
- Modes:
  - `--toy` — built-in synthetic dataset + tiny MLP (good for smoke tests)
  - `--entrypoint module.or.path:func` — dynamic loader for custom training targets
  - `--config config.json` — optional JSON dict passed to the entrypoint function

Entrypoint contract:
- The function referenced by `--entrypoint` must return a tuple `(model, train_dataset, eval_dataset_or_none)` compatible with PyTorch.

Examples:
```bash
# Built-in toy example
python -m medvllm.cli training train --toy --epochs 2 --batch-size 32 --lr 1e-3 --output ./toy_run

# Custom entrypoint from an installed module
python -m medvllm.cli training train \
  --entrypoint mypkg.my_module:build_training_objects \
  --config configs/train_small.json \
  --epochs 3 --batch-size 16 --lr 5e-4 --output ./runs/exp1

# Custom entrypoint from a local file path
python -m medvllm.cli training train \
  --entrypoint /abs/path/to/entrypoint.py:build \
  --config /abs/path/to/config.json \
  --epochs 1 --batch-size 8 --lr 1e-3 --output ./runs/path_exp
```

Notes:
- Minimal TPU support via torch_xla: use `--device xla`. Requires `torch-xla` installed and environment configured. AMP uses XLA autocast (bfloat16 recommended).

Further reading: `docs/TRAINER.md`.

## Return Codes
- `0` on success
- Non-zero with a human-friendly error message on failure (Click exceptions, import errors, invalid options)

## Tips
- Use `--json-out`/`--json-meta` for machine-readable outputs in pipelines.
- You can pipe input from other tools, e.g. `cat note.txt | python -m medvllm.cli inference ner --json-out`.

## Imaging and Perf Utilities (benchmarks)

Although not part of `medvllm.cli`, the repository provides imaging and attention benchmarking CLIs under `benchmarks/` that are useful during development and CI:

- `benchmarks/benchmark_imaging.py`
  - Depthwise Conv2D microbenchmark flags:
    - `--depthwise-bench` — run eager vs fused depthwise microbench.
    - `--depthwise-bench-iters N` — iterations per case (default 50).
    - `--depthwise-bench-sizes CxHxW[,CxHxW...]` — run multiple cases, e.g. `8x128x128,32x256x256`. Empty uses `--in-ch/--height/--width`.
  - Segmentation dataset regression:
    - `--seg-dataset seg2d_small` — evaluate Dice/IoU on a tiny 2D dataset (downloaded or synthetic fallback).
    - Env for CI enforcement (optional):
      - `MEDVLLM_SEG2D_URL` — dataset URL (enables enforcement when set)
      - `MEDVLLM_SEG2D_SHA256` — dataset checksum
      - `MEDVLLM_SEG_MIN_DICE` (default 0.70), `MEDVLLM_SEG_MIN_IOU` (default 0.55)
  - Profiling:
    - `--emit-trace` and `--trace-dir DIR` export Chrome traces via torch.profiler when available.

- `benchmarks/benchmark_attention.py`
  - Softmax×V microbenchmark flags:
    - `--attn-softmaxv-bench` — compare eager softmax×V vs fused Triton softmax×V (if available).
    - `--enable-triton-softmaxv` — sets `MEDVLLM_ENABLE_TRITON_SOFTMAXV=1` for this run to opt into the gated Triton path.
  - Common options:
    - `--seq`, `--heads`, `--dim`, `--batch`, `--iters`, `--device {auto|cpu|cuda}`, `--dtype {fp32,fp16,bf16}`
    - `--emit-trace`, `--trace-dir DIR` for profiler traces.
  - Example (CUDA):
    ```bash
    python benchmarks/benchmark_attention.py \
      --seq 512 --heads 8 --dim 64 --iters 50 \
      --device cuda --dtype bf16 \
      --attn-softmaxv-bench --enable-triton-softmaxv
    ```

See `docs/OPTIMIZATION.md` for more on fused kernels, profiling, and CI integration.

### Environment toggles (performance kernels)

- `MEDVLLM_DISABLE_TRITON=1` — force-disable Triton fused kernels (depthwise, etc.).
- `MEDVLLM_ENABLE_TRITON_SOFTMAXV=1` — opt-in to the Triton softmax×V placeholder path (disabled by default; may be slower than eager/SDPA).
- `MEDVLLM_ENABLE_TRITON_SEP3D=1` — opt-in to Triton depthwise 3D kernel (disabled by default).
- Tuning overrides for separable 3D (useful for sweeps):
  - `MEDVLLM_SEP3D_BLOCK_W` in {64,128,256}
  - `MEDVLLM_SEP3D_WARPS` in {2,4,8}
  - `MEDVLLM_SEP3D_STAGES` in {2,3,4}
- Fused options for separable 3D:
  - `MEDVLLM_SEP3D_FUSE_BIAS=1` — fuse bias add in-kernel
  - `MEDVLLM_SEP3D_FUSE_RELU=1` — fuse ReLU in-kernel

Example (CUDA):
```bash
MEDVLLM_ENABLE_TRITON_SEP3D=1 \
python benchmarks/benchmark_separable_conv3d.py
```
