"""Example: Using NERProcessor for medical NER.

This script demonstrates three ways to use NERProcessor:
1) Rule-based fallback (built-in RegexNERExtractor)
2) Plugging a model-backed pipeline exposing `run_inference(text, task_type="ner")`
3) A configurable gazetteer pipeline to extend/override dictionaries

Run:
  python examples/ner_processor_example.py

Outputs:
- Prints extracted entities, linked entities
- Writes HTML highlighting to examples/ner_highlight.html
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from medvllm.tasks import NERProcessor


# -----------------------------
# 2) Model-backed pipeline stub
# -----------------------------
class DummyModelPipeline:
    """A minimal model-backed pipeline example.

    It implements run_inference(text, task_type="ner") and returns
    a dict with an "entities" list. Replace with your real model code.
    """

    def __init__(self, entity_types: Iterable[str] | None = None) -> None:
        self.entity_types = [t.lower() for t in (entity_types or ["disease", "medication"])]

    def run_inference(self, text: str, task_type: str = "ner") -> Dict[str, Any]:
        if task_type != "ner":
            return {"entities": []}
        ents: List[Dict[str, Any]] = []
        # Extremely naive heuristic: if keywords appear, emit an entity span
        keywords = {
            "disease": ["myocardial infarction", "hypertension", "diabetes"],
            "medication": ["aspirin", "metformin"],
        }
        for et in self.entity_types:
            for kw in keywords.get(et, []):
                start = text.lower().find(kw)
                if start >= 0:
                    ents.append(
                        {
                            "text": text[start : start + len(kw)],
                            "type": et,
                            "start": start,
                            "end": start + len(kw),
                            "confidence": 0.85,
                        }
                    )
        ents.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        return {"entities": ents}


# ---------------------------------------
# 3) Configurable gazetteer-style pipeline
# ---------------------------------------
class GazetteerPipeline:
    """Regex gazetteer pipeline you can fully configure via a dictionary.

    gazetteer: dict[str, list[str]] mapping entity type to surface forms
    """

    def __init__(self, gazetteer: Dict[str, List[str]]) -> None:
        import re

        self._re = re
        self.patterns: list[tuple[str, Any]] = []
        for etype, words in gazetteer.items():
            items = sorted(set(words), key=len, reverse=True)
            if not items:
                continue
            pat = r"|".join(self._re.escape(w) for w in items)
            self.patterns.append(
                (etype.lower(), self._re.compile(rf"\b({pat})\b", self._re.IGNORECASE))
            )

    def run_inference(self, text: str, task_type: str = "ner") -> Dict[str, Any]:
        if task_type != "ner":
            return {"entities": []}
        ents: List[Dict[str, Any]] = []
        for et, rgx in self.patterns:
            for m in rgx.finditer(text):
                ents.append(
                    {
                        "text": m.group(0),
                        "type": et,
                        "start": m.start(),
                        "end": m.end(),
                        "confidence": 0.9,
                    }
                )
        ents.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        return {"entities": ents}


# -----------------------------
# Simple config holder (optional)
# -----------------------------
@dataclass
class SimpleConfig:
    # You can also use medvllm.medical.config.models.medical_config.MedicalModelConfig
    medical_entity_types: list[str] | None = None  # e.g. ["disease", "medication", ...]
    ner_enabled_entity_types: list[str] | None = None
    ner_type_hierarchy: Dict[str, str] | None = None
    ner_allow_unlisted_types: bool = False


# ---------------
# Demo functions
# ---------------
SAMPLE_TEXT = "Patient has myocardial infarction (MI). Aspirin given. Hemoglobin was low."


def demo_rule_based() -> None:
    print("\n[1] Rule-based fallback (built-in)")
    proc = NERProcessor(inference_pipeline=None, config=None)
    res = proc.extract_entities(SAMPLE_TEXT)
    print("EXTRACT:", json.dumps(res.to_dict(), indent=2))
    linked = proc.link_entities(res, ontology="UMLS")
    print("LINKED:", json.dumps(linked.to_dict(), indent=2))
    # Inline snippet (simple spans)
    html_inline = proc.highlight_entities(linked)
    out_inline = Path(__file__).with_name("ner_highlight.html")
    out_inline.write_text(html_inline, encoding="utf-8")
    print(f"HTML written to: {out_inline}")

    # Full-page interactive visualization with filters and details
    html_full = proc.highlight_entities(linked, format="html_full")
    out_full = Path(__file__).with_name("ner_visualization.html")
    out_full.write_text(html_full, encoding="utf-8")
    print(f"Interactive HTML written to: {out_full}")


def demo_model_backed() -> None:
    print("\n[2] Model-backed pipeline (dummy example)")
    pipeline = DummyModelPipeline(entity_types=["disease", "medication"])  # replace with real model
    proc = NERProcessor(
        inference_pipeline=pipeline,
        config=SimpleConfig(medical_entity_types=["disease", "medication"]),
    )
    res = proc.extract_entities(SAMPLE_TEXT)
    print("EXTRACT:", json.dumps(res.to_dict(), indent=2))
    linked = proc.link_entities(res, ontology="UMLS")
    print("LINKED:", json.dumps(linked.to_dict(), indent=2))


def demo_custom_gazetteer() -> None:
    print("\n[3] Custom gazetteer pipeline (configurable)")
    gazetteer = {
        "disease": ["myocardial infarction", "covid-19"],
        "medication": ["aspirin", "paracetamol"],
        "test": ["hemoglobin"],
    }
    pipeline = GazetteerPipeline(gazetteer)
    proc = NERProcessor(
        inference_pipeline=pipeline,
        config=SimpleConfig(medical_entity_types=list(gazetteer.keys())),
    )
    res = proc.extract_entities(SAMPLE_TEXT)
    print("EXTRACT:", json.dumps(res.to_dict(), indent=2))
    html = proc.highlight_entities(res)
    out = Path(__file__).with_name("ner_highlight_custom.html")
    out.write_text(html, encoding="utf-8")
    print(f"Custom HTML written to: {out}")


def demo_enable_disable_and_parent() -> None:
    print("\n[4] Enable/disable types and show parent_type")
    text = "Hemoglobin 13.5 g/dL measured on 2023-05-01. Aspirin given."
    # Enable only disease, lab_value, temporal
    cfg = SimpleConfig(
        medical_entity_types=["disease", "lab_value", "temporal", "medication", "test"],
        ner_enabled_entity_types=["disease", "lab_value", "temporal"],
    )
    proc = NERProcessor(inference_pipeline=None, config=cfg)
    res = proc.extract_entities(text)
    # Print compact view: type, parent_type, text
    compact = [
        {"type": e.get("type"), "parent_type": e.get("parent_type"), "text": e.get("text")}
        for e in res.entities
    ]
    print("FILTERED EXTRACT:", json.dumps(compact, indent=2))


def demo_allow_unlisted_types() -> None:
    print("\n[5] Allow unlisted types from a custom pipeline")

    class EmitsFooAndDisease:
        def run_inference(self, text: str, task_type: str = "ner") -> Dict[str, Any]:
            if task_type != "ner":
                return {"entities": []}
            ents: List[Dict[str, Any]] = []
            # disease span
            d = "myocardial infarction"
            ds = text.lower().find(d)
            if ds >= 0:
                ents.append(
                    {
                        "text": text[ds : ds + len(d)],
                        "type": "disease",
                        "start": ds,
                        "end": ds + len(d),
                        "confidence": 0.9,
                    }
                )
            # unlisted type span
            kw = "Aspirin"
            ks = text.find(kw)
            if ks >= 0:
                ents.append(
                    {
                        "text": text[ks : ks + len(kw)],
                        "type": "foo",
                        "start": ks,
                        "end": ks + len(kw),
                        "confidence": 0.8,
                    }
                )
            return {"entities": sorted(ents, key=lambda e: e["start"])}

    text = "Patient has myocardial infarction. Aspirin given."
    # Only disease known, but allow unlisted
    cfg = SimpleConfig(medical_entity_types=["disease"], ner_allow_unlisted_types=True)
    proc = NERProcessor(inference_pipeline=EmitsFooAndDisease(), config=cfg)
    res = proc.extract_entities(text)
    compact = [
        {
            "type": e.get("type"),
            "parent_type": e.get("parent_type"),
            "type_id": e.get("type_id"),
            "text": e.get("text"),
        }
        for e in res.entities
    ]
    print("ALLOW UNLISTED EXTRACT:", json.dumps(compact, indent=2))


if __name__ == "__main__":
    demo_rule_based()
    demo_model_backed()
    demo_custom_gazetteer()
    demo_enable_disable_and_parent()
    demo_allow_unlisted_types()
