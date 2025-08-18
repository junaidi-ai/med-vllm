"""NERProcessor: simple medical NER utilities.

This module provides a lightweight, dependency-free implementation to:
- extract entities (rule-based, regex/gazetteer)
- link entities to medical ontologies (stubbed)
- perform context-aware resolution (merge overlaps, resolve abbreviations)
- visualize entities (HTML highlighting)

It is designed to be easily swappable with a real model-backed pipeline later.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    # Optional: only for type hints if available
    from medvllm.medical.config.models.medical_config import MedicalModelConfig  # noqa: F401
except Exception:  # pragma: no cover - config not strictly required at runtime
    MedicalModelConfig = Any  # type: ignore


# -------------------------
# Data containers
# -------------------------
@dataclass
class NERResult:
    """Container for NER outputs returned by NERProcessor.

    entities: list of dicts with keys:
      - text: str
      - type: str
      - start: int (character index)
      - end: int (character index, exclusive)
      - confidence: float
      - optional: ontology_links: List[Dict[str, Any]]
    """

    text: str
    entities: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "entities": self.entities}


# -------------------------
# Simple ontology lookup stub
# -------------------------
_DEFAULT_ONTOLOGY_DB: Dict[str, Dict[str, Dict[str, Any]]] = {
    # ontology -> surface_form(lower) -> link info
    "UMLS": {
        "myocardial infarction": {"code": "C0027051", "name": "Myocardial Infarction"},
        "hypertension": {"code": "C0020538", "name": "Hypertensive disease"},
        "diabetes": {"code": "C0011849", "name": "Diabetes Mellitus"},
        "aspirin": {"code": "C0004057", "name": "Aspirin"},
    },
    "SNOMED": {
        "myocardial infarction": {"code": "22298006", "name": "Myocardial infarction"},
        "hypertension": {"code": "38341003", "name": "Hypertensive disorder"},
        "diabetes": {"code": "44054006", "name": "Diabetes mellitus"},
        "aspirin": {"code": "1191", "name": "Aspirin"},
    },
    "LOINC": {
        "hemoglobin": {"code": "718-7", "name": "Hemoglobin [Mass/volume] in Blood"},
    },
}


def lookup_in_ontology(text: str, entity_type: str, ontology: str = "UMLS") -> List[Dict[str, Any]]:
    """Very small, heuristic linking to a mock ontology DB.

    Returns a list of candidate links with simple scores.
    """
    out: List[Dict[str, Any]] = []
    key = text.strip().lower()
    ont = _DEFAULT_ONTOLOGY_DB.get(ontology.upper(), {})
    if key in ont:
        entry = ont[key]
        out.append(
            {
                "ontology": ontology.upper(),
                "code": entry["code"],
                "name": entry["name"],
                "score": 0.95,
                "type": entity_type,
            }
        )
    # Fallback: no match
    return out


# -------------------------
# Regex-based extractor
# -------------------------
class RegexNERExtractor:
    """Simple rule-based extractor using keyword regexes.

    This is a minimal, fast fallback when no model pipeline is provided.
    """

    def __init__(self, entity_types: Iterable[str]) -> None:
        self.entity_types = [str(t).lower() for t in entity_types]
        # Minimal gazetteer per type; extend as needed
        lex: Dict[str, List[str]] = {
            "disease": ["myocardial infarction", "hypertension", "diabetes"],
            "medication": ["aspirin", "metformin", "ibuprofen"],
            "procedure": ["angioplasty", "biopsy"],
            "symptom": ["chest pain", "fever", "cough"],
            "test": ["hemoglobin", "cbc"],
        }
        self.patterns: List[Tuple[str, re.Pattern[str]]] = []
        for et in self.entity_types:
            items = lex.get(et, [])
            if not items:
                continue
            # word-boundary, longest-first alternation
            items_sorted = sorted(items, key=len, reverse=True)
            pat = r"|".join(re.escape(w) for w in items_sorted)
            self.patterns.append((et, re.compile(rf"\b({pat})\b", flags=re.IGNORECASE)))

    def run_inference(self, text: str, task_type: str = "ner") -> Dict[str, Any]:
        if task_type != "ner":
            return {"entities": []}
        entities: List[Dict[str, Any]] = []
        for et, rgx in self.patterns:
            for m in rgx.finditer(text):
                entities.append(
                    {
                        "text": m.group(0),
                        "type": et,
                        "start": m.start(),
                        "end": m.end(),
                        "confidence": 0.9,
                    }
                )
        # sort by start
        entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        return {"entities": entities}


# -------------------------
# NER Processor
# -------------------------
class NERProcessor:
    """High-level convenience API for medical NER flows.

    Args:
        inference_pipeline: Object with a `run_inference(text, task_type="ner")` method
            returning a dict with key "entities": List[Dict[str, Any]]. If None, a
            RegexNERExtractor will be used based on config-provided entity types.
        config: Optional configuration carrying entity type information and linking settings.
                Recognized attributes:
                  - medical_entity_types or entity_types: Iterable[str]
                  - entity_linking.enabled: bool
    """

    def __init__(self, inference_pipeline: Optional[Any], config: Optional[Any]) -> None:
        self.config = config
        self.entity_types: List[str] = self._resolve_entity_types(config)
        self.pipeline = inference_pipeline or RegexNERExtractor(self.entity_types)
        # Type->id map for simple compatibility
        self._type_to_id = {t: i for i, t in enumerate(self.entity_types)}

    # ---- public API ----
    def extract_entities(self, text: str) -> NERResult:
        proc_text = self.preprocess_text(text)
        raw = self.pipeline.run_inference(proc_text, task_type="ner")
        entities = raw.get("entities", [])
        # ensure required fields and attach type_id
        norm: List[Dict[str, Any]] = []
        for ent in entities:
            et = str(ent.get("type", "other")).lower()
            norm.append(
                {
                    "text": ent.get("text", ""),
                    "type": et,
                    "start": int(ent.get("start", -1)),
                    "end": int(ent.get("end", -1)),
                    "confidence": float(ent.get("confidence", 0.0)),
                    "type_id": int(self._type_to_id.get(et, -1)),
                }
            )
        # context-aware pass: merge overlaps, resolve abbreviations
        norm = self._resolve_context(proc_text, norm)
        return NERResult(text=proc_text, entities=norm)

    def preprocess_text(self, text: str) -> str:
        """Preprocess input text before NER.

        Defaults to no-op to preserve character offsets. You can enable simple
        normalization via config flags on `self.config`:
        - ner_preprocess_collapse_whitespace: bool (default False)
        - ner_preprocess_lowercase: bool (default False)

        Note: Enabling these may alter character offsets relative to the original
        raw text. Use when your downstream pipeline re-aligns spans or does not
        rely on exact character indices of the raw input.
        """
        t = text
        try:
            collapse_ws = bool(getattr(self.config, "ner_preprocess_collapse_whitespace", False))
        except Exception:
            collapse_ws = False
        try:
            to_lower = bool(getattr(self.config, "ner_preprocess_lowercase", False))
        except Exception:
            to_lower = False

        if collapse_ws:
            t = re.sub(r"\s+", " ", t).strip()
        if to_lower:
            t = t.lower()
        return t

    def link_entities(self, ner_result: NERResult, ontology: str = "UMLS") -> NERResult:
        linked: List[Dict[str, Any]] = []
        for ent in ner_result.entities:
            links = lookup_in_ontology(ent["text"], ent["type"], ontology)
            new_ent = dict(ent)
            new_ent["ontology_links"] = links
            linked.append(new_ent)
        return NERResult(text=ner_result.text, entities=linked)

    def highlight_entities(self, ner_result: NERResult, format: str = "html") -> str:
        if format.lower() != "html":
            raise ValueError(
                "Only 'html' highlight format is supported in this simple implementation"
            )
        return self._to_html(ner_result)

    # ---- helpers ----
    def _resolve_entity_types(self, config: Optional[Any]) -> List[str]:
        # Prefer medical_entity_types from our config; fall back to entity_types or defaults
        if config is None:
            return ["disease", "medication", "procedure", "symptom", "test"]
        types = None
        for attr in ("medical_entity_types", "entity_types"):
            if hasattr(config, attr):
                types = getattr(config, attr)
                break
        if not types:
            return ["disease", "medication", "procedure", "symptom", "test"]
        # Normalize to lowercase strings
        return [str(t).lower() for t in (list(types) if not isinstance(types, list) else types)]

    def _resolve_context(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entities:
            return entities
        # 1) Merge overlapping or adjacent same-type entities, keep max confidence
        merged: List[Dict[str, Any]] = []
        for ent in entities:
            if not merged:
                merged.append(ent)
                continue
            last = merged[-1]
            if ent["type"] == last["type"] and ent["start"] <= last["end"] + 1:
                # merge
                last["end"] = max(last["end"], ent["end"])
                last["text"] = text[last["start"] : last["end"]]
                last["confidence"] = max(last.get("confidence", 0.0), ent.get("confidence", 0.0))
                continue
            merged.append(ent)
        # 2) Abbreviation resolution: "Long Form (ABBR)" -> link ABBR to Long Form
        # Simple regex to find patterns like "myocardial infarction (MI)"
        for m in re.finditer(
            r"(?P<long>[A-Za-z][A-Za-z\s/-]{2,}?)\s*\((?P<abbr>[A-Za-z]{2,8})\)", text
        ):
            long_start, long_end = m.start("long"), m.end("long")
            abbr_start, abbr_end = m.start("abbr"), m.end("abbr")
            # find entity covering long form
            covering = [e for e in merged if e["start"] <= long_start and e["end"] >= long_end]
            if not covering:
                continue
            base_ent = covering[0]
            # ensure abbreviation is present as entity of same type
            has_abbr = any(e["start"] == abbr_start and e["end"] == abbr_end for e in merged)
            if not has_abbr:
                merged.append(
                    {
                        "text": text[abbr_start:abbr_end],
                        "type": base_ent["type"],
                        "start": abbr_start,
                        "end": abbr_end,
                        "confidence": base_ent.get("confidence", 0.8) * 0.9,
                        "type_id": base_ent.get("type_id", -1),
                        "alias_of": base_ent["text"],
                    }
                )
        merged.sort(key=lambda e: (e["start"], e["end"]))
        return merged

    def _to_html(self, ner_result: NERResult) -> str:
        text = ner_result.text
        # Build non-overlapping spans; assume entities sorted by start
        spans: List[Tuple[int, int, Dict[str, Any]]] = []
        for e in sorted(ner_result.entities, key=lambda x: (x["start"], x["end"])):
            if e["start"] < 0 or e["end"] <= e["start"]:
                continue
            # Skip if overlaps last kept span
            if spans and e["start"] < spans[-1][1]:
                continue
            spans.append((e["start"], e["end"], e))

        # Generate HTML with simple inline styles; escape text
        out: List[str] = []
        cursor = 0
        palette = {
            "disease": "#fde2e1",
            "medication": "#e1f5fe",
            "procedure": "#e8f5e9",
            "symptom": "#fff3e0",
            "test": "#ede7f6",
        }
        for start, end, e in spans:
            if cursor < start:
                out.append(html.escape(text[cursor:start]))
            label = e.get("type", "entity")
            bg = palette.get(label, "#f0f0f0")
            title = f"{label} (conf={e.get('confidence', 0.0):.2f})"
            out.append(
                f"<span style=\"background:{bg};border-radius:4px;padding:1px 3px;\" title=\"{html.escape(title)}\">{html.escape(text[start:end])}</span>"
            )
            cursor = end
        if cursor < len(text):
            out.append(html.escape(text[cursor:]))
        return "".join(out)
