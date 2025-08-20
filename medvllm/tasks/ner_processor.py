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
# Entity type system (hierarchy + toggles)
# -------------------------
class EntityTypeSystem:
    """Minimal entity type registry with hierarchy and enable/disable toggles.

    Config (all optional):
      - ner_type_hierarchy: Dict[str, Optional[str]] mapping type -> parent
      - ner_enabled_entity_types: Iterable[str] to explicitly enable a subset
      - medical_entity_types / entity_types: Iterable[str] (back-compat)

    If no config is provided, sensible defaults are used.
    """

    def __init__(self, config: Optional[Any]) -> None:
        # Defaults
        default_types = [
            "disease",
            "medication",
            "procedure",
            "symptom",
            "test",
            "anatomical_structure",
            "lab_value",
            "temporal",
        ]
        # Simple default hierarchy (single-parent)
        default_hierarchy: Dict[str, Optional[str]] = {
            "disease": "clinical_finding",
            "symptom": "clinical_finding",
            "medication": "treatment",
            "procedure": "treatment",
            "test": "observation",
            "lab_value": "observation",
            "anatomical_structure": "entity",
            "temporal": "metadata",
            # abstract parents
            "clinical_finding": "entity",
            "treatment": "entity",
            "observation": "entity",
            "metadata": None,
            "entity": None,
        }

        enabled: List[str] = []
        hierarchy: Dict[str, Optional[str]] = dict(default_hierarchy)

        # Back-compat: entity type list hints
        hinted_types: Optional[Iterable[str]] = None
        if config is not None:
            for attr in ("medical_entity_types", "entity_types"):
                if hasattr(config, attr):
                    hinted_types = getattr(config, attr)
                    break
        if hinted_types:
            enabled = [str(t).lower() for t in list(hinted_types)]
        else:
            enabled = list(default_types)

        # Optional explicit hierarchy override from config
        if config is not None and hasattr(config, "ner_type_hierarchy"):
            try:
                cfg_h = getattr(config, "ner_type_hierarchy")
                if isinstance(cfg_h, dict):
                    # Normalize to lower-case keys/values
                    normalized: Dict[str, Optional[str]] = {}
                    for k, v in cfg_h.items():
                        normalized[str(k).lower()] = None if v is None else str(v).lower()
                    hierarchy.update(normalized)
            except Exception:
                pass

        # Optional explicit enabled types override from config
        if config is not None and hasattr(config, "ner_enabled_entity_types"):
            try:
                ets = getattr(config, "ner_enabled_entity_types")
                enabled = [str(t).lower() for t in list(ets)]
            except Exception:
                pass

        # Build type->id map from enabled leaf- and mid-level types (order stable)
        self.enabled_types: List[str] = list(dict.fromkeys(enabled))
        self.hierarchy: Dict[str, Optional[str]] = hierarchy
        self.type_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.enabled_types)}

    def parent_of(self, t: str) -> Optional[str]:
        return self.hierarchy.get(t)

    def is_a(self, t: str, ancestor: str) -> bool:
        t = t.lower()
        ancestor = ancestor.lower()
        seen = set()
        while t is not None and t not in seen:
            if t == ancestor:
                return True
            seen.add(t)
            t = self.hierarchy.get(t)
        return False


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
        "pneumonia": {"code": "C0032310", "name": "Pneumonia"},
        "asthma": {"code": "C0004096", "name": "Asthma"},
        "stroke": {"code": "C0038454", "name": "Cerebrovascular accident"},
        "covid-19": {"code": "C5203670", "name": "COVID-19"},
        "metformin": {"code": "C0025598", "name": "Metformin"},
        "ibuprofen": {"code": "C0020740", "name": "Ibuprofen"},
        "atorvastatin": {"code": "C0717701", "name": "Atorvastatin"},
        "lisinopril": {"code": "C0070374", "name": "Lisinopril"},
        "insulin": {"code": "C0021641", "name": "Insulins"},
    },
    "SNOMED": {
        "myocardial infarction": {"code": "22298006", "name": "Myocardial infarction"},
        "hypertension": {"code": "38341003", "name": "Hypertensive disorder"},
        "diabetes": {"code": "44054006", "name": "Diabetes mellitus"},
        "aspirin": {"code": "1191", "name": "Aspirin"},
        "pneumonia": {"code": "233604007", "name": "Pneumonia"},
        "asthma": {"code": "195967001", "name": "Asthma"},
        "stroke": {"code": "230690007", "name": "Cerebrovascular accident"},
        "covid-19": {"code": "840539006", "name": "Disease caused by SARS-CoV-2"},
    },
    "LOINC": {
        "hemoglobin": {"code": "718-7", "name": "Hemoglobin [Mass/volume] in Blood"},
        "cbc": {"code": "57021-8", "name": "CBC panel - Blood"},
    },
    "RXNORM": {
        "aspirin": {"code": "1191", "name": "Aspirin"},
        "metformin": {"code": "6809", "name": "Metformin"},
        "ibuprofen": {"code": "5640", "name": "Ibuprofen"},
        "atorvastatin": {"code": "83367", "name": "Atorvastatin"},
        "lisinopril": {"code": "29046", "name": "Lisinopril"},
        "insulin": {"code": "6042", "name": "Insulin"},
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

    def __init__(self, entity_types: Iterable[str], config: Optional[Any] = None) -> None:
        self.entity_types = [str(t).lower() for t in entity_types]
        self.config = config
        # Minimal gazetteer per type; extend as needed
        lex: Dict[str, List[str]] = {
            "disease": [
                "myocardial infarction",
                "hypertension",
                "diabetes",
                "pneumonia",
                "asthma",
                "stroke",
                "covid-19",
            ],
            "medication": [
                "aspirin",
                "metformin",
                "ibuprofen",
                "atorvastatin",
                "lisinopril",
                "insulin",
            ],
            "procedure": [
                "angioplasty",
                "biopsy",
                "ct scan",
                "mri",
                "x-ray",
            ],
            "symptom": [
                "chest pain",
                "fever",
                "cough",
                "shortness of breath",
                "headache",
                "nausea",
            ],
            "test": [
                "hemoglobin",
                "cbc",
                "wbc",
                "platelet",
                "blood pressure",
                "hdl",
                "ldl",
            ],
            "anatomical_structure": [
                "heart",
                "liver",
                "kidney",
                "left ventricle",
                "right atrium",
                "lung",
                "brain",
                "pancreas",
                "stomach",
            ],
        }
        # Optional extended gazetteer via config flags
        enable_ext = False
        try:
            enable_ext = bool(getattr(self.config, "ner_enable_extended_gazetteer", False))
        except Exception:
            enable_ext = False
        if enable_ext:
            # Add a few extra representative items
            lex.setdefault("disease", []).extend(["sepsis", "copd", "ckd", "anemia"])
            lex.setdefault("medication", []).extend(["amoxicillin", "warfarin", "heparin"])
            lex.setdefault("procedure", []).extend(["echocardiogram", "colonoscopy"])
            lex.setdefault("symptom", []).extend(["fatigue", "dizziness"])
            lex.setdefault("test", []).extend(["troponin", "creatinine"])
            lex.setdefault("anatomical_structure", []).extend(["aorta", "spleen"])

        # Custom gazetteer from config: Dict[str, List[str]]
        try:
            custom_lex = getattr(self.config, "ner_custom_lexicon", None)
            if isinstance(custom_lex, dict):
                for k, vs in custom_lex.items():
                    if not isinstance(vs, (list, tuple)):
                        continue
                    lex.setdefault(str(k).lower(), []).extend([str(v).lower() for v in vs])
        except Exception:
            pass

        # Special regexes for certain types (not pure gazetteer)
        special_patterns: Dict[str, str] = {
            # e.g., "Hemoglobin 13.5 g/dL", "Na 137 mmol/L", "Hb 11%", "TSH 3.2 mIU/L"
            "lab_value": r"\b([A-Za-z]{1,4}[A-Za-z\s/]*)\s*(\d+(?:\.\d+)?)\s*(g/dL|mg/dL|mmol/L|mEq/L|IU/L|mIU/L|U/L|IU/mL|mIU/mL|mg/L|mcg/mL|ng/mL|g/L|%)\b",
            # Dates like 2023-05-01 or 05/01/2023, and relative times like '2 days ago'
            "temporal": r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago)\b",
        }

        self.patterns: List[Tuple[str, re.Pattern[str]]] = []
        for et in self.entity_types:
            if et in special_patterns:
                self.patterns.append((et, re.compile(special_patterns[et], flags=re.IGNORECASE)))
                continue
            items = lex.get(et, [])
            if not items:
                # No lexicon and no special regex; skip
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
                  - ner_enabled_entity_types: Iterable[str] (explicit enable list)
                  - ner_type_hierarchy: Dict[str, Optional[str]] (type -> parent)
                  - entity_linking.enabled: bool
                  - ner_allow_unlisted_types: bool (default False) — if True, do not filter
                    model outputs to the enabled types list.
    """

    def __init__(self, inference_pipeline: Optional[Any], config: Optional[Any]) -> None:
        self.config = config
        # Build type system (handles defaults, hierarchy, enable toggles)
        self.type_system = EntityTypeSystem(config)
        self.entity_types: List[str] = self.type_system.enabled_types
        self.pipeline = inference_pipeline or RegexNERExtractor(self.entity_types, config)
        # Type->id map for simple compatibility
        self._type_to_id = dict(self.type_system.type_to_id)
        try:
            self._allow_unlisted = bool(getattr(self.config, "ner_allow_unlisted_types", False))
        except Exception:
            self._allow_unlisted = False
        # Optional confidence threshold
        try:
            thr = getattr(self.config, "ner_confidence_threshold", None)
            self._conf_threshold: Optional[float] = float(thr) if thr is not None else None
        except Exception:
            self._conf_threshold = None

    # ---- public API ----
    def extract_entities(self, text: str) -> NERResult:
        proc_text = self.preprocess_text(text)
        raw = self.pipeline.run_inference(proc_text, task_type="ner")
        entities = raw.get("entities", [])
        # ensure required fields and attach type_id
        norm: List[Dict[str, Any]] = []
        enabled_types = set(self.type_system.enabled_types)

        def _is_enabled_or_descendant(t: str) -> bool:
            if t in enabled_types:
                return True
            # If a parent type is enabled, allow its descendants
            for anc in enabled_types:
                if self.type_system.is_a(t, anc):
                    return True
            return False

        for ent in entities:
            et = str(ent.get("type", "other")).lower()
            conf = float(ent.get("confidence", 0.0))
            if self._conf_threshold is not None and conf < self._conf_threshold:
                continue
            if not self._allow_unlisted and not _is_enabled_or_descendant(et):
                continue
            parent = self.type_system.parent_of(et)
            norm.append(
                {
                    "text": ent.get("text", ""),
                    "type": et,
                    "start": int(ent.get("start", -1)),
                    "end": int(ent.get("end", -1)),
                    "confidence": conf,
                    "type_id": int(self._type_to_id.get(et, -1)),
                    "parent_type": parent,
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
            # leaf/common types
            "disease": "#fde2e1",
            "medication": "#e1f5fe",
            "procedure": "#e8f5e9",
            "symptom": "#fff3e0",
            "test": "#ede7f6",
            "anatomical_structure": "#f0f4c3",
            "lab_value": "#d7ccc8",
            "temporal": "#c8e6c9",
            # abstract parents (in case upstream emits them)
            "clinical_finding": "#ffcdd2",
            "treatment": "#bbdefb",
            "observation": "#d1c4e9",
            "entity": "#f5f5f5",
            "metadata": "#cfd8dc",
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
