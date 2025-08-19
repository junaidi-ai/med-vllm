# NERProcessor

A lightweight, pluggable utility for medical named entity recognition (NER).

- Extraction via either:
  - A provided model-backed pipeline exposing `run_inference(text, task_type="ner")`
  - A built-in regex/gazetteer fallback (no extra dependencies)
- Post-processing:
  - Merge overlapping/adjacent same-type entities
  - Simple abbreviation resolution: "Long Form (ABBR)"
- Entity linking (stub): link to simple mock ontologies (UMLS/SNOMED/LOINC)
- Visualization: HTML highlighting with minimal inline styling

## Import

```python
from medvllm.tasks import NERProcessor, NERResult
```

## Basic Usage (rule-based fallback)

```python
proc = NERProcessor(inference_pipeline=None, config=None)
text = "Patient has myocardial infarction (MI). Aspirin given. Hemoglobin was low."
res = proc.extract_entities(text)        # NERResult
linked = proc.link_entities(res, "UMLS") # NERResult with ontology_links per entity
html = proc.highlight_entities(linked)    # HTML string
```

## Model-backed Pipeline

Provide any object with `run_inference(text, task_type="ner") -> {"entities": [...]}`.

Entity items should be dicts with: `text`, `type`, `start`, `end`, `confidence`.

```python
class MyModelPipeline:
    def run_inference(self, text: str, task_type: str = "ner") -> dict:
        # return {"entities": [...]} from your model
        ...

pipeline = MyModelPipeline()
proc = NERProcessor(inference_pipeline=pipeline, config=None)
res = proc.extract_entities("...")
```

## Configuring Entity Types

`NERProcessor` determines entity types from `config` if available:

- `config.medical_entity_types` or `config.entity_types` should be an iterable of strings, e.g.,
  `['disease', 'medication', 'procedure', 'symptom', 'test']`.
- If `config` is `None` or missing fields, defaults are used.

This controls the regex fallback lexicons and the type->id mapping.

## Preprocessing Hook

`NERProcessor.preprocess_text(text)` runs before extraction.

- Defaults: no-op (to preserve offsets)
- Optional flags on `config`:
  - `ner_preprocess_collapse_whitespace: bool` (default False)
  - `ner_preprocess_lowercase: bool` (default False)

Note: enabling these may alter character offsets relative to the raw input.

## Example Script

See `examples/ner_processor_example.py` for:

- Rule-based fallback
- Model-backed pipeline stub
- Configurable gazetteer-style pipeline

## Notes

- Ontology linking here is a stub for demonstration.
- HTML highlighting aims to be simple and dependency-free.
