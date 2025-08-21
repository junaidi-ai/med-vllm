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
import json
from urllib import request as urllib_request, error as urllib_error, parse as urllib_parse
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

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


# Fuzzy similarity acceptance threshold (global, cached). Updated via setter.
_FUZZY_THRESHOLD: float = 0.5


def set_fuzzy_threshold(value: float) -> None:
    """Set global fuzzy similarity threshold and clear lookup cache.

    Threshold is clamped to [0.0, 1.0]. Changing this clears the LRU cache so
    that subsequent lookups reflect the new acceptance threshold.
    """
    global _FUZZY_THRESHOLD
    try:
        v = float(value)
    except Exception:
        return
    v = max(0.0, min(1.0, v))
    if v != _FUZZY_THRESHOLD:
        _FUZZY_THRESHOLD = v
        lookup_in_ontology.cache_clear()


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
# Simple ontology lookup (enhanced stub)
# -------------------------
# ontology -> surface_form(lower) -> link info
_DEFAULT_ONTOLOGY_DB: Dict[str, Dict[str, Dict[str, Any]]] = {
    "UMLS": {
        "myocardial infarction": {
            "code": "C0027051",
            "name": "Myocardial Infarction",
            "synonyms": ["mi", "heart attack"],
        },
        "hypertension": {"code": "C0020538", "name": "Hypertensive disease", "synonyms": []},
        "diabetes": {"code": "C0011849", "name": "Diabetes Mellitus", "synonyms": ["dm"]},
        "aspirin": {"code": "C0004057", "name": "Aspirin", "synonyms": ["acetylsalicylic acid"]},
        "pneumonia": {"code": "C0032310", "name": "Pneumonia", "synonyms": []},
        "asthma": {"code": "C0004096", "name": "Asthma", "synonyms": []},
        "stroke": {"code": "C0038454", "name": "Cerebrovascular accident", "synonyms": ["cva"]},
        "covid-19": {"code": "C5203670", "name": "COVID-19", "synonyms": ["sars-cov-2 infection"]},
        "metformin": {"code": "C0025598", "name": "Metformin", "synonyms": ["metformin hcl"]},
        "ibuprofen": {"code": "C0020740", "name": "Ibuprofen", "synonyms": []},
        "atorvastatin": {"code": "C0717701", "name": "Atorvastatin", "synonyms": []},
        "lisinopril": {"code": "C0070374", "name": "Lisinopril", "synonyms": []},
        "insulin": {"code": "C0021641", "name": "Insulins", "synonyms": []},
    },
    "SNOMED": {
        "myocardial infarction": {
            "code": "22298006",
            "name": "Myocardial infarction",
            "synonyms": ["mi"],
        },
        "hypertension": {"code": "38341003", "name": "Hypertensive disorder", "synonyms": []},
        "diabetes": {"code": "44054006", "name": "Diabetes mellitus", "synonyms": []},
        "aspirin": {"code": "1191", "name": "Aspirin", "synonyms": []},
        "pneumonia": {"code": "233604007", "name": "Pneumonia", "synonyms": []},
        "asthma": {"code": "195967001", "name": "Asthma", "synonyms": []},
        "stroke": {"code": "230690007", "name": "Cerebrovascular accident", "synonyms": []},
        "covid-19": {"code": "840539006", "name": "Disease caused by SARS-CoV-2", "synonyms": []},
    },
    "LOINC": {
        "hemoglobin": {
            "code": "718-7",
            "name": "Hemoglobin [Mass/volume] in Blood",
            "synonyms": ["hb"],
        },
        "cbc": {
            "code": "57021-8",
            "name": "CBC panel - Blood",
            "synonyms": ["complete blood count"],
        },
    },
    "RXNORM": {
        "aspirin": {"code": "1191", "name": "Aspirin", "synonyms": ["asa"]},
        "metformin": {
            "code": "6809",
            "name": "Metformin",
            "synonyms": ["metformin hcl", "glucophage"],
        },
        "ibuprofen": {"code": "5640", "name": "Ibuprofen", "synonyms": []},
        "atorvastatin": {"code": "83367", "name": "Atorvastatin", "synonyms": []},
        "lisinopril": {"code": "29046", "name": "Lisinopril", "synonyms": []},
        "insulin": {"code": "6042", "name": "Insulin", "synonyms": []},
    },
}


def _normalize_surface(text: str) -> str:
    t = text.strip().lower()
    # Replace separators with spaces and collapse
    t = re.sub(r"[\-/_,]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> Set[str]:
    return set(_normalize_surface(text).split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _ontology_uri(ontology: str, code: str) -> str:
    o = ontology.upper()
    if o == "UMLS":
        return f"https://uts.nlm.nih.gov/umls/concept/{code}"
    if o == "SNOMED":
        return f"http://snomed.info/id/{code}"
    if o == "LOINC":
        return f"https://loinc.org/{code}/"
    if o == "RXNORM":
        return f"https://rxnav.nlm.nih.gov/REST/rxcui/{code}"
    return f"{o}:{code}"


@lru_cache(maxsize=1024)
def lookup_in_ontology(
    text: str, entity_type: str, ontology: str = "UMLS"
) -> Tuple[Dict[str, Any], ...]:
    """Heuristic linking to a small in-memory ontology DB with simple fuzzy matching.

    Notes:
    - Cached via LRU to speed up repeated lookups.
    - Returns a tuple of link dicts; callers should convert to list if needed.
    """
    ont_name = ontology.upper()
    ont = _DEFAULT_ONTOLOGY_DB.get(ont_name, {})
    norm = _normalize_surface(text)

    # 1) Exact match on canonical keys
    if norm in ont:
        entry = ont[norm]
        link = {
            "ontology": ont_name,
            "code": entry["code"],
            "name": entry["name"],
            "score": 0.95,
            "type": entity_type,
            "uri": _ontology_uri(ont_name, entry["code"]),
        }
        return (link,)

    # 2) Synonym and simple fuzzy matching (token Jaccard)
    cand_tokens = _tokenize(norm)
    best: Optional[Tuple[float, str, Dict[str, Any]]] = None  # (score, key, entry)
    for key, entry in ont.items():
        # compare against canonical key
        key_tokens = _tokenize(key)
        score = _jaccard(cand_tokens, key_tokens)
        # also compare against synonyms
        for syn in entry.get("synonyms", []) or []:
            syn_tokens = _tokenize(syn)
            score = max(score, _jaccard(cand_tokens, syn_tokens))
        if best is None or score > best[0]:
            best = (score, key, entry)

    # Accept if similarity above a (configurable) threshold
    if best and best[0] >= _FUZZY_THRESHOLD:
        _, _key, entry = best
        link = {
            "ontology": ont_name,
            "code": entry["code"],
            "name": entry["name"],
            "score": float(min(0.9, 0.6 + best[0] * 0.4)),
            "type": entity_type,
            "uri": _ontology_uri(ont_name, entry["code"]),
        }
        return (link,)

    # 3) No match
    return ()


def _http_get_json(url: str, timeout: float = 3.0) -> Dict[str, Any]:
    """Tiny helper to GET JSON with stdlib. Returns {} on failure."""
    try:
        req = urllib_request.Request(url, headers={"User-Agent": "medvllm-ner/0.1"})
        with urllib_request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            data = resp.read()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {}


def _http_post_form(
    url: str, form: Dict[str, str], timeout: float = 3.0
) -> Tuple[Dict[str, Any], bytes, Dict[str, Any]]:
    """POST application/x-www-form-urlencoded. Returns (headers, body_bytes, info_dict).

    info_dict is a minimal mapping with keys like 'status' and 'location' if present.
    """
    data = urllib_parse.urlencode(form).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={
            "User-Agent": "medvllm-ner/0.1",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            body = resp.read()
            hdrs = dict(resp.headers.items())
            info = {
                "status": getattr(resp, "status", None) or resp.getcode(),
                "location": hdrs.get("Location"),
            }
            return hdrs, body, info
    except Exception as e:
        return {}, b"", {"error": str(e)}


def _umls_request_tgt(api_key: str, timeout: float = 3.0) -> Optional[str]:
    """Obtain a UMLS TGT URL using API key.

    Docs: https://documentation.uts.nlm.nih.gov/rest/authentication.html
    POST to https://utslogin.nlm.nih.gov/cas/v1/api-key with apikey=KEY.
    On success, Location header contains the TGT URL; alternatively, the response
    HTML form's action attribute carries the TGT URL.
    """
    tgt_endpoint = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
    hdrs, body, info = _http_post_form(tgt_endpoint, {"apikey": api_key}, timeout=timeout)
    # Prefer Location header
    loc = info.get("location") if isinstance(info, dict) else None
    if loc:
        return str(loc)
    # Fallback: parse HTML 'action' attribute
    try:
        s = body.decode("utf-8", errors="ignore")
        m = re.search(r'action="([^"]*?/TGT-[^"]+)"', s)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _umls_request_service_ticket(tgt_url: str, service: str, timeout: float = 3.0) -> Optional[str]:
    """Obtain a one-time service ticket from a TGT URL for the given service."""
    _, body, info = _http_post_form(tgt_url, {"service": service}, timeout=timeout)
    if isinstance(info, dict) and info.get("error"):
        return None
    try:
        ticket = body.decode("utf-8").strip()
        return ticket or None
    except Exception:
        return None


def _umls_fetch_cui(cui: str, ticket: str, timeout: float = 3.0) -> Dict[str, Any]:
    """Fetch UMLS concept by CUI using a valid service ticket."""
    url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{urllib_parse.quote(cui)}?ticket={urllib_parse.quote(ticket)}"
    try:
        req = urllib_request.Request(url, headers={"User-Agent": "medvllm-ner/0.1"})
        with urllib_request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            data = resp.read()
        js = json.loads(data.decode("utf-8"))
        return js if isinstance(js, dict) else {}
    except Exception:
        return {}


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
        # Optional fuzzy threshold for ontology linking
        try:
            el = getattr(self.config, "entity_linking", None)
            if el is not None and hasattr(el, "fuzzy_threshold"):
                set_fuzzy_threshold(float(getattr(el, "fuzzy_threshold")))
        except Exception:
            pass

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
        # Respect optional config gating and default ontology selection
        ont = ontology
        enabled = True
        try:
            el = getattr(self.config, "entity_linking", None)
            if el is not None and hasattr(el, "enabled"):
                enabled = bool(getattr(el, "enabled"))
            # default ontology from config if provided
            if el is not None and hasattr(el, "default_ontology"):
                ont = str(getattr(el, "default_ontology")) or ont
        except Exception:
            pass
        if not enabled:
            return NERResult(text=ner_result.text, entities=list(ner_result.entities))

        linked: List[Dict[str, Any]] = []
        for ent in ner_result.entities:
            links_tuple = lookup_in_ontology(ent["text"], ent["type"], ont)
            new_ent = dict(ent)
            # expose as list to callers
            new_ent["ontology_links"] = list(links_tuple)
            linked.append(new_ent)
        return NERResult(text=ner_result.text, entities=linked)

    def fetch_link_details(self, link: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch additional details for a given ontology link.

        Gated by config: requires `config.entity_linking.external.enabled`.
        - RXNORM: queries RxNav public REST (no API key required).
        - UMLS: requires API key. If `entity_linking.external.umls_cas_enabled` is True,
          performs the CAS/TGT authentication flow and fetches the CUI metadata from UTS.
          Otherwise (default), returns a placeholder note to avoid network calls during tests.

        Returns a dictionary with fetched details or None if disabled/unsupported.
        """
        try:
            el = getattr(self.config, "entity_linking", None)
            ext = getattr(el, "external", None) if el is not None else None
            enabled = bool(getattr(ext, "enabled", False)) if ext is not None else False
            if not enabled:
                return None
            timeout = float(getattr(ext, "timeout", 3.0)) if ext is not None else 3.0
            umls_api_key = getattr(ext, "umls_api_key", None) if ext is not None else None
        except Exception:
            return None

        ontology = str(link.get("ontology", "")).upper()
        code = str(link.get("code", "")).strip()
        if not ontology or not code:
            return None

        if ontology == "RXNORM":
            url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{code}/properties.json"
            data = _http_get_json(url, timeout=timeout)
            props = data.get("properties") if isinstance(data, dict) else None
            if props:
                return {"source": "RXNORM", "properties": props}
            return {}

        if ontology == "UMLS":
            # Requires UMLS UTS API key and CAS ticket flow.
            if not umls_api_key:
                return None
            # Optional CAS flow gating to preserve backward-compatible behavior in tests
            cas_enabled = False
            try:
                cas_enabled = bool(getattr(ext, "umls_cas_enabled", False))
            except Exception:
                cas_enabled = False
            if not cas_enabled:
                return {
                    "source": "UMLS",
                    "note": (
                        "UMLS CAS flow not implemented in this lightweight client. "
                        "Set entity_linking.external.umls_cas_enabled=True to enable CAS/TGT, "
                        "or use UTS authentication externally to fetch concept metadata for code: "
                        + code
                    ),
                }
            # Proceed with CAS/TGT flow
            tgt_url = _umls_request_tgt(str(umls_api_key), timeout=timeout)
            if not tgt_url:
                return {"source": "UMLS", "error": "umls_auth_failed", "note": "TGT request failed"}
            service = "http://umlsks.nlm.nih.gov"
            try:
                svc = getattr(ext, "umls_service", None)
                if svc:
                    service = str(svc)
            except Exception:
                pass
            st = _umls_request_service_ticket(tgt_url, service, timeout=timeout)
            if not st:
                return {
                    "source": "UMLS",
                    "error": "umls_ticket_failed",
                    "note": "Service ticket request failed",
                }
            data = _umls_fetch_cui(code, st, timeout=timeout)
            if data:
                # UTS wraps the result in {"result": {...}}
                result = data.get("result", data)
                return {"source": "UMLS", "result": result}
            return {}

        return None

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
        # ---- New context-aware annotations ----
        # Helper: sentence spans (very simple segmentation)
        sent_spans: List[Tuple[int, int]] = []
        s_start = 0
        for m in re.finditer(r"[.!?;\n]", text):
            s_end = m.start()
            if s_end > s_start:
                sent_spans.append((s_start, s_end))
            s_start = m.end()
        if s_start < len(text):
            sent_spans.append((s_start, len(text)))

        def find_sentence_span(pos: int) -> Tuple[int, int]:
            for a, b in sent_spans:
                if a <= pos < b:
                    return a, b
            return 0, len(text)

        # Collect temporal entities for later attachment
        temporal_ents = [e for e in merged if e.get("type") == "temporal"]

        # Negation and modifier cues (simple heuristics)
        neg_cues = re.compile(
            r"\b(?:no|denies|denied|without|negative for|free of|ruled out|rule out|absent|not|neither|nor)\b",
            flags=re.IGNORECASE,
        )
        severity_cues = re.compile(
            r"\b(mild|moderate|severe|slight|marked|significant)\b", re.IGNORECASE
        )
        certainty_cues = re.compile(
            r"\b(possible|probable|likely|unlikely|suspicious for|concern for|suggestive of|consistent with)\b",
            re.IGNORECASE,
        )
        temporal_rel_cues = [
            (re.compile(r"\bsince\b", re.IGNORECASE), "since"),
            (re.compile(r"\bfor\b", re.IGNORECASE), "for"),
            (re.compile(r"\bon\b", re.IGNORECASE), "on"),
            (re.compile(r"\bafter\b", re.IGNORECASE), "after"),
            (re.compile(r"\bbefore\b", re.IGNORECASE), "before"),
            (re.compile(r"\buntil\b", re.IGNORECASE), "until"),
            (re.compile(r"\bby\b", re.IGNORECASE), "by"),
        ]

        def nearest_temporal(e: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not temporal_ents:
                return None
            # Prefer same sentence, else nearest by char distance within 80 chars
            a, b = find_sentence_span(e["start"])
            same_sent = [t for t in temporal_ents if a <= t["start"] < b]
            cands = same_sent or temporal_ents
            best = None
            best_d = 10**9
            for t in cands:
                d = min(abs(e["start"] - t["end"]), abs(t["start"] - e["end"]))
                if d < best_d:
                    best, best_d = t, d
            if best is not None and best_d <= 80:
                return best
            return None

        # Annotate each entity (except temporals themselves) with negation, modifiers, temporal link
        for e in merged:
            etype = str(e.get("type", "")).lower()
            s_a, s_b = find_sentence_span(e["start"]) if e.get("start", -1) >= 0 else (0, len(text))
            # Context window before entity within sentence
            pre_ctx_start = max(s_a, e["start"] - 60)
            pre_ctx = text[pre_ctx_start : e["start"]]
            # Negation for relevant clinical types
            if etype in {"disease", "symptom", "test", "clinical_finding"}:
                tail = pre_ctx[-30:].lower()
                if neg_cues.search(tail):
                    e["negated"] = True
            # Modifiers
            mods: Dict[str, Any] = {}
            m_sev = severity_cues.search(pre_ctx)
            if m_sev:
                mods["severity"] = m_sev.group(1).lower()
            m_cert = certainty_cues.search(pre_ctx)
            if m_cert:
                # Normalize multi-word matches
                mods["certainty"] = m_cert.group(0).lower()
            if mods:
                e["modifiers"] = mods
            # Temporal association
            if etype != "temporal":
                t_ent = nearest_temporal(e)
                if t_ent is not None:
                    e["temporal"] = t_ent.get("text")
                    e["temporal_span"] = (t_ent.get("start"), t_ent.get("end"))
                    # Temporal relation cue in pre-context
                    for rx, label in temporal_rel_cues:
                        if rx.search(pre_ctx):
                            e["temporal_relation"] = label
                            break

        # Basic coreference grouping: identical normalized surface (or alias_of target)
        def norm_key(e: Dict[str, Any]) -> Tuple[str, str]:
            base = str(e.get("alias_of") or e.get("text") or "")
            return (str(e.get("type", "")).lower(), _normalize_surface(base))

        groups: Dict[Tuple[str, str], int] = {}
        gid = 1
        for e in merged:
            key = norm_key(e)
            if not key[1]:
                continue
            if key not in groups:
                groups[key] = gid
                gid += 1
            e["coref_group"] = groups[key]
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
