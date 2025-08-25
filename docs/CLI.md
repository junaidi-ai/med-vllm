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
    └── inference
        ├── ner
        ├── generate
        └── classification
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

Examples:
```bash
# Patient-friendly summary with beam strategy and metadata JSON
python -m medvllm.cli inference generate \
  --text "Explain hypertension to a patient." \
  --model your-hf-model \
  --strategy beam --beam-width 3 \
  --purpose patient --readability general --tone informal \
  --target-words 120 --json-meta

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

## Return Codes
- `0` on success
- Non-zero with a human-friendly error message on failure (Click exceptions, import errors, invalid options)

## Tips
- Use `--json-out`/`--json-meta` for machine-readable outputs in pipelines.
- You can pipe input from other tools, e.g. `cat note.txt | python -m medvllm.cli inference ner --json-out`.
