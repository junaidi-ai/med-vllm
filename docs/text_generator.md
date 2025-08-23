# TextGenerator: Length and Style Controls

This document describes the extended controls for text generation in `medvllm/tasks/text_generator.py`.

## New Controls (all optional)
- `readability`: audience guidance. Examples: `"general"`, `"specialist"`.
- `tone`: writing tone. Examples: `"formal"`, `"informal"`, `"friendly"`.
- `structure`: response structure. Examples: `"soap"`, `"bullet"`, `"paragraph"`.
- `specialty`: domain focus. Example: `"cardiology"`, `"oncology"`, `"pulmonology"`.
- `target_words`: soft word cap enforced after generation (adds `...` if truncated).
- `target_chars`: hard character cap via `MedicalConstraints.max_length_chars`.
- `style_preset`: JSON path or dict to apply multiple fields at once.

These are additive to the existing `style` argument (e.g., `"formal"`, `"concise"`, `"patient_friendly"`).

## Usage Examples

```python
from medvllm.tasks import TextGenerator, MedicalConstraints

# Initialize with your engine (LLM or FakeEngine in examples)
tg = TextGenerator(engine, constraints=MedicalConstraints())

# Fine-grained controls
res = tg.generate(
    "Summarize hypertension management.",
    strategy="beam",
    max_length=128,
    readability="general",
    tone="formal",
    structure="soap",
    specialty="cardiology",
    target_words=120,
    target_chars=800,
)
print(res.generated_text)
print(res.metadata)

# Load a style preset from JSON
res2 = tg.generate(
    "Patient education on diabetes foot care.",
    strategy="greedy",
    style_preset="examples/presets/patient_education.json",
)
```

## Style Presets
- Save and load presets via `TextGenerator.save_style_preset(path, preset_dict)` and `TextGenerator.load_style_preset(path)`.
- Preset keys are shallowly applied to `MedicalConstraints`. Recognized keys include:
  - `banned_phrases`, `required_disclaimer`, `enforce_disclaimer`, `max_length_chars`
  - `hipaa_filter_enabled`, `ontology_verify_enabled`, `ontology_default`
  - `factual_checks_enabled`, `contradiction_detection_enabled`, `speculative_flagging_enabled`, `confidence_scoring_enabled`
  - `add_inline_citations`, `append_references_section`, `hedging_markers`
  - `expand_abbreviations`, `abbreviation_map`, `normalize_units_spacing`
  - `profile`, `readability_level`, `tone`, `structure`, `specialty`, `target_word_count`

## Notes
- Defaults preserve existing behavior when new arguments are not provided.
- `target_words` is a soft cap; final text may be slightly shorter/longer.
- `target_chars` applies before the disclaimer is appended.
```
