# User Acceptance Testing (UAT)

This document describes manual, scriptable UAT scenarios for Med vLLM's CLI.

- Location of scripts: `scripts/uat/`
  - Driver: `run_uat.sh`
  - Scenarios: `uat_cli_basic.sh`, `uat_cli_inference.sh`
  - Golden expectations: `scripts/uat/expected/`

## Prerequisites

- Python environment with project installed (editable install recommended):

```bash
pip install -r requirements/requirements-dev.txt
pip install -e .
```

- Recommended environment variables for stability:

```bash
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM="false"
```

## Running all UAT scenarios

```bash
bash scripts/uat/run_uat.sh
```

The driver runs all scenario scripts and prints a summary. Exit code is non‑zero if any scenario fails.

## Scenarios

- `uat_cli_basic.sh`
  - `python -m medvllm.cli examples` — verifies help text includes quick examples.
  - `python -m medvllm.cli model list` — verifies default registry prints entries.

- `uat_cli_inference.sh`
  - `python -m medvllm.cli inference ner --text "HTN on metformin" --json-out` — verifies JSON-like keys present.
  - `python -m medvllm.cli inference generate --model gpt2 --output <file>` with prompt from stdin — verifies the output file is created and non-empty.

## Golden expectations format

Golden files under `scripts/uat/expected/` are line-based checks:

- Regular line: required substring in stdout.
- `OR: a || b || c`: at least one of the alternatives must appear in stdout.
- `FILE_NONEMPTY: <path>`: the given path must exist and be non-empty.

## CI integration

You can integrate UAT into CI by adding a step/job to run:

```bash
bash scripts/uat/run_uat.sh
```

Failures will cause the job to fail, providing a simple acceptance gate.
