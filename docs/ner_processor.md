# NERProcessor

A lightweight, pluggable utility for medical named entity recognition (NER).

- Extraction via either:
  - A provided model-backed pipeline exposing `run_inference(text, task_type="ner")`
  - A built-in regex/gazetteer fallback (no extra dependencies)
- Post-processing:
  - Merge overlapping/adjacent same-type entities
  - Simple abbreviation resolution: "Long Form (ABBR)"
- Entity linking (stub): link to simple mock ontologies (UMLS/SNOMED/LOINC/RXNORM)
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

If you are using the built-in `MedicalNER` model wrapper, you can use the provided adapter:

```python
from medvllm.tasks import MedicalNER, MedicalNERAdapter, NERProcessor

model = MedicalNER.load_pretrained("biobert-base-cased-v1.2")  # example
adapter = MedicalNERAdapter(model, config=my_config)
proc = NERProcessor(inference_pipeline=adapter, config=my_config)
res = proc.extract_entities("Patient given aspirin. TSH 3.2 mIU/L.")
```

## Supported Entity Types

Out of the box, the rule-based fallback recognizes these types:

- disease
- medication
- procedure
- symptom
- test
- anatomical_structure
- lab_value (e.g., "Hemoglobin 13.5 g/dL")
- temporal (e.g., "2023-05-01", "2 days ago")

You can add your own via a custom pipeline.

The built-in gazetteer includes representative examples (e.g., diseases like pneumonia/asthma/stroke, medications like metformin/atorvastatin/lisinopril/insulin, procedures like CT scan/MRI, symptoms like shortness of breath/headache) but is intentionally minimal for speed and can be extended.

## Configuring Entity Types

`NERProcessor` determines entity types from `config` if available:

- `config.medical_entity_types` or `config.entity_types` should be an iterable of strings, e.g.,
  `['disease', 'medication', 'procedure', 'symptom', 'test', 'anatomical_structure', 'lab_value', 'temporal']`.
- `config.ner_enabled_entity_types` can explicitly enable a subset regardless of the hints above. Filtering respects hierarchy (see below): enabling a parent (e.g., `treatment`) allows its children (e.g., `medication`, `procedure`) to pass filtering.
- `config.ner_allow_unlisted_types` (default False): if True, do not filter model-emitted types to the enabled list/hierarchy.
- `config.ner_confidence_threshold` (float in [0,1]): entities with `confidence` below this are dropped during extraction.
- If `config` is `None` or missing fields, defaults are used.

This controls the regex fallback lexicons and the type->id mapping.

### Extended/Custom Gazetteer

The regex/gazetteer fallback can be extended via config flags:

- `config.ner_enable_extended_gazetteer: bool` — adds extra representative items across diseases, medications, procedures, symptoms, tests, and anatomical structures.
- `config.ner_custom_lexicon: Dict[str, List[str]]` — provide your own items per type key, e.g., `{ "medication": ["apixaban", "clopidogrel"], "test": ["CRP"] }`.

Lab value patterns include common units: `g/dL`, `mg/dL`, `mmol/L`, `mEq/L`, `IU/L`, `mIU/L`, `U/L`, `IU/mL`, `mIU/mL`, `mg/L`, `mcg/mL`, `ng/mL`, `g/L`, `%`.

## Hierarchy (optional)

You can optionally declare a simple single-parent hierarchy for types by setting
`config.ner_type_hierarchy: Dict[str, Optional[str]]` where keys are types and values
are parent types (or `None`). Defaults include abstract parents:

- clinical_finding, treatment, observation, metadata, entity

Use `EntityTypeSystem.is_a(child, ancestor)` internally to reason about relationships. The extractor's filtering allows entities whose type is a descendant of any enabled parent type. Note: the built-in regex fallback only emits patterns for the explicitly enabled leaf types; custom/model pipelines can emit any type and will be filtered using the hierarchy.

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

- Ontology linking here is a stub for demonstration; it includes small demonstration dictionaries for UMLS, SNOMED, LOINC, and RXNORM.
- HTML highlighting aims to be simple and dependency-free.
