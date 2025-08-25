# NERProcessor

A lightweight, pluggable utility for medical named entity recognition (NER).

- Extraction via either:
  - A provided model-backed pipeline exposing `run_inference(text, task_type="ner")`
  - A built-in regex/gazetteer fallback (no extra dependencies)
- Post-processing:
  - Merge overlapping/adjacent same-type entities
  - Simple abbreviation resolution: "Long Form (ABBR)"
- Entity linking: links to small in-memory mock ontologies (UMLS/SNOMED/LOINC/RXNORM), with simple fuzzy matching and caching
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

## Entity Linking

`NERProcessor.link_entities(ner_result, ontology="UMLS")` attaches `ontology_links` to each entity.

- Config gating: set `config.entity_linking.enabled = False` to disable linking (default: True if unspecified).
- Default ontology: set `config.entity_linking.default_ontology = "RXNORM"` (or other) to change the default used when `ontology` arg is omitted.
- Fuzzy matching: a lightweight token-based Jaccard similarity is used with synonym support from the mock KBs (e.g., "metformin hcl" matches RXNORM Metformin).
- Caching: lookups are memoized via an LRU cache for repeated queries.
- Link fields: each link includes `ontology`, `code`, `name`, `score`, `type`, `uri`.

Notes:
- This is still an in-memory demonstration KB for offline use; integrate real APIs (e.g., UMLS, RxNav) in production.

### External Ontology Enrichment (optional)

You can fetch additional metadata for an attached link via `NERProcessor.fetch_link_details(link)`.

- Gate via `config.entity_linking.external.enabled = True`.
- Optional timeout: `config.entity_linking.external.timeout` (seconds).
- RxNorm: Uses public RxNav REST (no API key required). Example:

```python
cfg = SimpleNamespace(entity_linking=SimpleNamespace(external=SimpleNamespace(enabled=True)))
proc = NERProcessor(inference_pipeline=None, config=cfg)
details = proc.fetch_link_details({"ontology": "RXNORM", "code": "1191"})  # Aspirin
```

- UMLS: Requires a UMLS API key. Two modes:
  - Default (no CAS): returns a placeholder note to avoid network calls in tests.
  - CAS/TGT enabled: set `config.entity_linking.external.umls_cas_enabled = True` and provide
    `config.entity_linking.external.umls_api_key`. Optionally override `umls_service`
    (default `http://umlsks.nlm.nih.gov`). The client uses stdlib `urllib` to perform the CAS flow.

```python
from types import SimpleNamespace
cfg = SimpleNamespace(
    entity_linking=SimpleNamespace(
        external=SimpleNamespace(
            enabled=True,
            umls_api_key="<YOUR_UMLS_API_KEY>",
            umls_cas_enabled=True,
            timeout=3.0,
            # umls_service="http://umlsks.nlm.nih.gov",  # optional override
        )
    )
)
proc = NERProcessor(inference_pipeline=None, config=cfg)
link = {"ontology": "UMLS", "code": "C0004057"}
details = proc.fetch_link_details(link)
```

Notes:
- The UMLS integration uses the documented CAS/TGT ticket workflow via `utslogin.nlm.nih.gov` and
  queries UTS for the CUI. If the CAS flag is disabled (default), a placeholder message is returned.
- This remains a lightweight, dependency-free client; consider a dedicated SDK for production needs.

## Extended/Custom Gazetteer

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

## Benchmarking Linking Cache

Use the included benchmark to measure linking timing and cache hits on long notes:

```bash
python3 -m benchmarks.benchmark_linking --paragraphs 50 --runs 3 --ontology RXNORM
```

It reports extract time, each link run timing, and `lookup_in_ontology()` cache hits/misses.

## Notes

- Ontology linking here is a stub for demonstration; it includes small demonstration dictionaries for UMLS, SNOMED, LOINC, and RXNORM.
- HTML highlighting aims to be simple and dependency-free.

## CLI Usage (Dual-Mode)

`NERProcessor` powers the CLI `inference ner` command in two modes:

- Processor-only (default): runs without model dependencies.
- Model-backed: pass `--model <name>` to use a Hugging Face token-classification pipeline internally via an adapter.

Validation: when `--model` is provided, the CLI checks the model registry metadata and ensures `capabilities["tasks"]` includes `"ner"`. If unsupported, the command fails; if the model is unregistered, validation is skipped with a warning.

Output schema is consistent across modes (per-entity): `text`, `type`, `start`, `end`, optional `confidence`, and optional `ontology_links` (unless `--no-link`). See `docs/CLI.md` for examples.
