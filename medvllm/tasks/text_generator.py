"""Medical text generation utilities.

Provides a simple `TextGenerator` that wraps the med-vLLM engine (`LLM`) and
adds:
- generation strategies: greedy, sampling, simple multi-sample "beam"
- top-p and top-k sampling via a logits processor
- basic medical constraints filtering and optional disclaimer
- simple factual consistency checker stub

This module is intentionally lightweight and self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import logging

import torch
import json
import os

from medvllm.llm import LLM
from medvllm.sampling_params import SamplingParams
from .generation_strategies import create_strategy
from .generation_backends import create_backend

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    prompt: str
    generated_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MedicalConstraints:
    banned_phrases: List[str] = field(default_factory=list)
    required_disclaimer: Optional[str] = (
        "This is informational and not a substitute for professional medical advice."
    )
    enforce_disclaimer: bool = True
    # Optional simple length/content constraints
    max_length_chars: Optional[int] = None
    # HIPAA/PII filtering (very lightweight regex-based)
    hipaa_filter_enabled: bool = True
    # Ontology verification of medical terms (uses NERProcessor stubs)
    ontology_verify_enabled: bool = True
    ontology_default: str = "UMLS"
    # Factual consistency analysis (opt-in outputs for citations/references)
    factual_checks_enabled: bool = True
    contradiction_detection_enabled: bool = True
    speculative_flagging_enabled: bool = True
    confidence_scoring_enabled: bool = True
    add_inline_citations: bool = False
    append_references_section: bool = False
    # Hedging and absolute markers for simple heuristics
    hedging_markers: List[str] = field(
        default_factory=lambda: [
            "may",
            "might",
            "could",
            "possibly",
            "possible",
            "suggests",
            "appears",
            "likely",
            "unlikely",
            "potential",
            "consider",
        ]
    )
    # Abbreviation handling (expand common medical abbreviations)
    expand_abbreviations: bool = False
    abbreviation_map: Dict[str, str] = field(
        default_factory=lambda: {
            "MI": "myocardial infarction",
            "HTN": "hypertension",
            "DM": "diabetes mellitus",
            "CVA": "stroke",
            "BP": "blood pressure",
            "HR": "heart rate",
            "ASA": "aspirin",
            "Hb": "hemoglobin",
        }
    )
    # Formatting rules
    normalize_units_spacing: bool = True
    # Profile controls (purpose-specific presets): 'patient', 'clinical', 'research'
    profile: Optional[str] = None
    # Readability, tone, structure and specialty styling toggles
    readability_level: Optional[str] = None  # 'general' | 'specialist'
    tone: Optional[str] = None  # 'formal' | 'informal'
    structure: Optional[str] = None  # 'soap' | 'bullet' | 'paragraph'
    specialty: Optional[str] = None  # e.g., 'cardiology', 'oncology'
    # Length controls
    target_word_count: Optional[int] = None

    def apply_output_filters(self, text: str) -> str:
        # Redact banned phrases in a simple way
        for phrase in self.banned_phrases:
            if not phrase:
                continue
            text = text.replace(phrase, "[REDACTED]")
        # Truncate to max length if specified
        if self.max_length_chars is not None and self.max_length_chars > 0:
            text = text[: self.max_length_chars]
        # Ensure disclaimer presence
        if self.enforce_disclaimer and self.required_disclaimer:
            if self.required_disclaimer not in text:
                # Append with separation
                sep = "\n\n" if not text.endswith("\n") else "\n"
                text = f"{text}{sep}{self.required_disclaimer}"
        return text

    # --- extended filters/utilities ---
    def apply_profile_overrides(self, purpose: Optional[str]) -> None:
        """Adjust constraint toggles based on purpose profile.

        - patient: strong HIPAA, expand abbreviations, keep disclaimer
        - clinical: strong HIPAA, no disclaimer append, do not expand abbreviations
        - research: moderate HIPAA, keep disclaimer optional
        """
        if not purpose:
            return
        p = str(purpose).strip().lower()
        self.profile = p
        if p in {"patient", "patient_communication", "patient-friendly"}:
            self.hipaa_filter_enabled = True
            self.expand_abbreviations = True
            self.enforce_disclaimer = True
        elif p in {"clinical", "clinical_notes", "notes"}:
            self.hipaa_filter_enabled = True
            self.expand_abbreviations = False
            # Clinical notes typically don't include disclaimers in the body
            self.enforce_disclaimer = False
        elif p in {"research", "paper", "publication"}:
            self.hipaa_filter_enabled = True
            self.expand_abbreviations = False
            # Disclaimer optional
            self.enforce_disclaimer = False

    # --- presets ---
    def apply_preset(self, preset: Dict[str, Any]) -> None:
        """Shallow update of constraint-related fields from a preset dict."""
        if not isinstance(preset, dict):
            return
        # recognized keys
        keys = {
            "banned_phrases",
            "required_disclaimer",
            "enforce_disclaimer",
            "max_length_chars",
            "hipaa_filter_enabled",
            "ontology_verify_enabled",
            "ontology_default",
            "factual_checks_enabled",
            "contradiction_detection_enabled",
            "speculative_flagging_enabled",
            "confidence_scoring_enabled",
            "add_inline_citations",
            "append_references_section",
            "hedging_markers",
            "expand_abbreviations",
            "abbreviation_map",
            "normalize_units_spacing",
            "profile",
            "readability_level",
            "tone",
            "structure",
            "specialty",
            "target_word_count",
        }
        for k in keys:
            if k in preset:
                setattr(self, k, preset[k])

    def _hipaa_redact(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Very simple PII redaction using regex; returns (text, counts).

        Patterns covered:
        - Dates (YYYY-MM-DD, MM/DD/YYYY)
        - Phone numbers
        - Email addresses
        - SSN-like numbers
        - MRN-like IDs (alphanumeric 6-10 chars)
        - Addresses (very naive: numbers + street)
        """
        import re

        counts: Dict[str, int] = {
            "date": 0,
            "phone": 0,
            "email": 0,
            "ssn": 0,
            "mrn": 0,
            "address": 0,
        }
        s = text

        def sub_count(pattern: str, repl: str, key: str) -> None:
            nonlocal s
            found = re.findall(pattern, s)
            if not found:
                return
            counts[key] += len(found)
            s = re.sub(pattern, repl, s)

        sub_count(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b", "[DATE]", "date")
        sub_count(
            r"\b(?:\+?\d{1,2}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}\b", "[PHONE]", "phone"
        )
        sub_count(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]", "email")
        sub_count(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", "ssn")
        # MRN-like IDs: 6-10 upper alnum chars with at least one digit
        sub_count(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{6,10}\b", "[ID]", "mrn")
        sub_count(
            r"\b\d+\s+[A-Za-z]+\s+(Street|St|Ave|Avenue|Rd|Road|Blvd|Lane|Ln)\b",
            "[ADDRESS]",
            "address",
        )
        return s, counts

    def _expand_abbr(self, text: str) -> Tuple[str, Dict[str, int]]:
        import re

        counts: Dict[str, int] = {"expanded": 0}
        s = text
        for abbr, long_form in self.abbreviation_map.items():
            # If already present as Long (ABBR), skip; otherwise, expand first occurrence
            if re.search(
                rf"\b{re.escape(long_form)}\s*\(\s*{re.escape(abbr)}\s*\)", s, flags=re.IGNORECASE
            ):
                continue
            # Expand standalone ABBR not part of a larger word
            pattern = rf"\b{re.escape(abbr)}\b"
            if re.search(pattern, s):
                s = re.sub(pattern, f"{long_form} ({abbr})", s, count=1)
                counts["expanded"] += 1
        return s, counts

    def _format_units(self, text: str) -> Tuple[str, Dict[str, int]]:
        import re

        units = ["mg", "g", "mcg", "kg", "mL", "mmHg", "bpm"]
        s = text
        changes = 0
        for u in units:
            # Ensure a space between number and unit (e.g., 10mg -> 10 mg)
            # Do not break composite units like mg/dL
            pattern = rf"(\d)(?:\s?){re.escape(u)}\b"
            repl = rf"\\1 {u}"
            new_s = re.sub(pattern, repl, s)
            if new_s != s:
                diff = len(re.findall(pattern, s))
                changes += diff
                s = new_s
        return s, {"unit_spacing_normalized": changes}

    def verify_via_ontology(self, text: str, ontology: Optional[str] = None) -> Dict[str, Any]:
        """Use the lightweight NERProcessor to find and link medical entities.

        Returns a report dict with linked entities and any entities that failed to link
        above the fuzzy threshold.
        """
        if not self.ontology_verify_enabled:
            return {"enabled": False}
        try:
            # Lazy import to avoid heavy deps during test collection
            from .ner_processor import NERProcessor
        except Exception:
            return {"enabled": True, "error": "ner_processor_unavailable"}

        try:
            proc = NERProcessor(inference_pipeline=None, config=None)
            ner = proc.extract_entities(text)
            linked = proc.link_entities(ner, ontology=ontology or self.ontology_default)
            linked_list = []
            unverified: List[Dict[str, Any]] = []
            for e in linked.entities:
                links = e.get("ontology_links") or []
                if links:
                    # only surface a compact summary
                    link0 = links[0]
                    linked_list.append(
                        {
                            "text": e.get("text"),
                            "type": e.get("type"),
                            "code": link0.get("code"),
                            "ontology": link0.get("ontology"),
                            "name": link0.get("name"),
                            "score": link0.get("score"),
                            "uri": link0.get("uri"),
                        }
                    )
                else:
                    unverified.append({"text": e.get("text"), "type": e.get("type")})
            return {"enabled": True, "linked": linked_list, "unverified": unverified}
        except Exception as e:
            return {"enabled": True, "error": str(e)}


class TextGenerator:
    def __init__(
        self,
        model_or_engine: Any,
        constraints: Optional[MedicalConstraints] = None,
        **engine_kwargs: Any,
    ) -> None:
        """Initialize the medical text generator.

        Args:
            model_or_engine: Hugging Face model ID/path or an existing LLM engine.
            constraints: Optional MedicalConstraints to apply to outputs.
            **engine_kwargs: Extra kwargs passed to `LLM` if a model string is given.
        """
        self.constraints = constraints or MedicalConstraints()
        # Accept any engine-like object that implements .generate()
        if hasattr(model_or_engine, "generate"):
            self.engine = model_or_engine  # type: ignore[assignment]
        elif isinstance(model_or_engine, str):
            # Lazily construct LLM engine from model name/path
            self.engine = LLM(model_or_engine, **engine_kwargs)
        else:
            raise TypeError(
                "model_or_engine must be an engine-like object with .generate() or a model name/path string"
            )

    # -----------------------------
    # Public API
    # -----------------------------
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        strategy: str = "beam",
        temperature: float = 0.7,
        top_p: Optional[float] = 0.9,
        top_k: Optional[int] = None,
        beam_width: int = 3,
        style: Optional[str] = None,
        ignore_eos: bool = False,
        # runtime configuration for adapters/strategies
        backend: str = "gpt",
        template: Optional[str] = None,
        template_vars: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Any]] = None,
        # purpose/profile for constraints
        purpose: Optional[str] = None,
        # new: fine-grained styling & length controls
        readability: Optional[str] = None,
        tone: Optional[str] = None,
        structure: Optional[str] = None,
        specialty: Optional[str] = None,
        target_words: Optional[int] = None,
        target_chars: Optional[int] = None,
        # simple JSON style preset support (path or dict)
        style_preset: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> GenerationResult:
        """Generate text according to the strategy and constraints.

        Strategies:
        - greedy: temperature=0.0
        - sampling: temperature>0 with optional top_p/top_k
        - beam: simple multi-sample selection of N candidates (not true beam search)
        """
        # Apply purpose/profile overrides on constraints if provided
        try:
            self.constraints.apply_profile_overrides(purpose)
        except Exception:
            pass

        # Apply (optional) preset first
        preset_meta: Dict[str, Any] = {}
        if style_preset is not None:
            try:
                if isinstance(style_preset, str):
                    loaded = self.load_style_preset(style_preset)
                    if isinstance(loaded, dict):
                        self.constraints.apply_preset(loaded)
                        preset_meta["loaded_preset_path"] = style_preset
                elif isinstance(style_preset, dict):
                    self.constraints.apply_preset(style_preset)
                    preset_meta["loaded_preset_inline"] = True
            except Exception as e:
                preset_meta["preset_error"] = str(e)

        # Override constraint length by characters if specified
        if target_chars is not None and target_chars > 0:
            self.constraints.max_length_chars = target_chars
        # Keep target words inside constraints for metadata, but enforce post-gen
        if target_words is not None and target_words > 0:
            self.constraints.target_word_count = target_words
        # Adjust readability/tone/structure/specialty at constraint level for metadata
        if readability:
            self.constraints.readability_level = readability
        if tone:
            self.constraints.tone = tone
        if structure:
            self.constraints.structure = structure
        if specialty:
            self.constraints.specialty = specialty

        styled_prompt = self._apply_style_components(
            prompt,
            style=style,
            readability=self.constraints.readability_level,
            tone=self.constraints.tone,
            structure=self.constraints.structure,
            specialty=self.constraints.specialty,
            target_words=self.constraints.target_word_count,
        )
        backend_adapter = create_backend(backend)
        prepared_prompt = backend_adapter.prepare_prompt(styled_prompt)

        # Configure logits processor for top-p/top-k if requested
        logits_processor = self._build_logits_processor(top_p=top_p, top_k=top_k)
        self._set_logits_processor(logits_processor)

        # Strategy selection
        strat = create_strategy(
            strategy,
            template=template,
            template_vars=template_vars,
            few_shot_examples=few_shot_examples,
        )

        def runner(p: str, sp: SamplingParams) -> str:
            return self._run_once(p, sp)

        text = strat.generate(
            prepared_prompt,
            runner,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            beam_width=beam_width,
            ignore_eos=ignore_eos,
        )

        # Meta retains 0 temp for greedy
        effective_temperature = 0.0 if strategy.lower() == "greedy" else temperature

        # Apply output-side constraints in stages
        post_text = backend_adapter.postprocess(text)
        # Enforce target word count softly after generation (before other filters)
        if (
            self.constraints.target_word_count is not None
            and self.constraints.target_word_count > 0
        ):
            try:
                post_text = self._truncate_by_words(post_text, self.constraints.target_word_count)
            except Exception:
                pass
        constraints_report: Dict[str, Any] = {}

        # HIPAA redaction
        redacted_text = post_text
        if getattr(self.constraints, "hipaa_filter_enabled", False):
            try:
                redacted_text, hipaa_counts = self.constraints._hipaa_redact(redacted_text)
                constraints_report["hipaa"] = hipaa_counts
            except Exception as e:
                constraints_report["hipaa_error"] = str(e)

        # Abbreviation expansion (e.g., patient-friendly)
        expanded_text = redacted_text
        if getattr(self.constraints, "expand_abbreviations", False):
            try:
                expanded_text, abbr_counts = self.constraints._expand_abbr(expanded_text)
                constraints_report["abbreviations"] = abbr_counts
            except Exception as e:
                constraints_report["abbr_error"] = str(e)

        # Formatting rules
        formatted_text = expanded_text
        if getattr(self.constraints, "normalize_units_spacing", False):
            try:
                formatted_text, fmt_counts = self.constraints._format_units(formatted_text)
                constraints_report["formatting"] = fmt_counts
            except Exception as e:
                constraints_report["formatting_error"] = str(e)

        # Ontology verification on the text after PHI redaction and formatting
        try:
            ont_report = self.constraints.verify_via_ontology(
                formatted_text, ontology=self.constraints.ontology_default
            )
            constraints_report["ontology"] = ont_report
        except Exception as e:
            constraints_report["ontology_error"] = str(e)

        # Optional factual analysis (citations, references, contradictions, confidence)
        factual_report: Dict[str, Any] = {}
        to_filter_text = formatted_text
        if getattr(self.constraints, "factual_checks_enabled", False):
            try:
                factual_report = self._analyze_factual_consistency(formatted_text)
                # Inline citations
                if getattr(self.constraints, "add_inline_citations", False):
                    cited = factual_report.get("text_with_citations")
                    if isinstance(cited, str) and cited:
                        to_filter_text = cited
                # Append references section before disclaimer
                if getattr(self.constraints, "append_references_section", False):
                    refs = factual_report.get("references") or []
                    if isinstance(refs, list) and refs:
                        to_filter_text = self._append_references_section(to_filter_text, refs)
            except Exception as e:
                factual_report = {"error": str(e)}

        # Finally, apply generic content filters (banned phrases, disclaimer, etc.)
        filtered = self.constraints.apply_output_filters(to_filter_text)

        meta: Dict[str, Any] = {
            "strategy": strategy,
            "temperature": effective_temperature,
            "top_p": top_p,
            "top_k": top_k,
            "beam_width": beam_width,
            "style": style,
            "purpose": purpose,
            "readability": self.constraints.readability_level,
            "tone": self.constraints.tone,
            "structure": self.constraints.structure,
            "specialty": self.constraints.specialty,
            "target_words": self.constraints.target_word_count,
            "target_chars": self.constraints.max_length_chars,
            "constraints_report": constraints_report,
            "factual_report": factual_report,
        }
        if preset_meta:
            meta["style_preset"] = preset_meta

        return GenerationResult(prompt=prompt, generated_text=filtered, metadata=meta)

    def generate_with_context(
        self,
        prompt: str,
        context: str,
        **kwargs: Any,
    ) -> GenerationResult:
        combined_input = f"Context: {context}\n\nPrompt: {prompt}"
        return self.generate(combined_input, **kwargs)

    def check_medical_accuracy(self, generated_text: str) -> Dict[str, Any]:
        """Very simple heuristic checker. Returns a report dict.

        Note: This is a placeholder; integrate with a KB or verifier model later.
        """
        issues: List[str] = []
        lower = generated_text.lower()
        risky_markers = ["always", "never", "guarantee", "cure", "miracle"]
        for m in risky_markers:
            if m in lower:
                issues.append(f"Contains absolute/overclaim marker: '{m}'")
        for phrase in self.constraints.banned_phrases:
            if phrase and phrase.lower() in lower:
                issues.append(f"Contains banned phrase: '{phrase}'")
        ok = len(issues) == 0
        return {"ok": ok, "issues": issues}

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _apply_style(self, prompt: str, style: Optional[str]) -> str:
        # Backward-compatible wrapper
        return self._apply_style_components(prompt, style=style)

    def _apply_style_components(
        self,
        prompt: str,
        *,
        style: Optional[str] = None,
        readability: Optional[str] = None,
        tone: Optional[str] = None,
        structure: Optional[str] = None,
        specialty: Optional[str] = None,
        target_words: Optional[int] = None,
    ) -> str:
        """Compose an instruction prefix capturing style controls in a simple way."""
        lines: List[str] = []
        # Legacy style presets
        if style:
            s = style.lower()
            if s == "concise":
                lines.append("Please respond concisely and to the point.")
            elif s == "formal":
                lines.append("Use a formal, clinical tone suitable for medical documentation.")
            elif s in {"patient_friendly", "patient-friendly", "lay"}:
                lines.append("Explain in plain language suitable for patients. Avoid jargon.")
            else:
                lines.append(f"Style: {style}. Write accordingly.")

        # Readability
        if readability:
            r = readability.lower()
            if r in {"general", "patient", "public"}:
                lines.append("Audience: general public. Avoid technical jargon; define terms.")
            elif r in {"specialist", "professional", "clinician"}:
                lines.append("Audience: medical professionals. Use precise clinical terminology.")

        # Tone
        if tone:
            t = tone.lower()
            if t in {"formal", "clinical"}:
                lines.append("Tone: formal, objective, and clinically precise.")
            elif t in {"informal", "friendly"}:
                lines.append("Tone: conversational and approachable while remaining accurate.")

        # Structure templates
        if structure:
            st = structure.lower()
            if st in {"soap", "soap_notes", "soap-note"}:
                lines.append(
                    "Structure the response as SOAP notes: Subjective, Objective, Assessment, Plan."
                )
            elif st in {"bullet", "bulleted"}:
                lines.append("Use concise bullet points where appropriate.")
            elif st in {"paragraph", "narrative"}:
                lines.append("Use cohesive paragraphs with clear transitions.")

        # Specialty-specific styling
        if specialty:
            lines.append(
                f"Specialty focus: {specialty}. Prefer terminology and nuances for this domain."
            )

        # Target length (words)
        if target_words is not None and target_words > 0:
            lines.append(f"Target length: about {int(target_words)} words (concise if possible).")

        if lines:
            prefix = "\n".join(lines) + "\n\n"
            return prefix + prompt
        return prompt

    def _truncate_by_words(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        truncated = " ".join(words[:max_words]).rstrip()
        # Preserve closing punctuation if it exists shortly after cutoff
        if not truncated.endswith(('.', '!', '?')):
            truncated += "..."
        return truncated

    def _run_once(self, prompt: str, sp: SamplingParams) -> str:
        outputs = self.engine.generate([prompt], sp, use_tqdm=False)
        if not outputs:
            return ""
        text = outputs[0].get("text", "")
        return text

    def _select_best(self, candidates: Iterable[str]) -> str:
        # Simple heuristic: prefer first non-empty, otherwise longest
        cands = [c or "" for c in candidates]
        for c in cands:
            if c.strip():
                return c
        return max(cands, key=len) if cands else ""

    def _build_logits_processor(
        self, top_p: Optional[float], top_k: Optional[int]
    ) -> Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]]:
        if top_p is None and top_k is None:
            return None

        def processor(logits: torch.Tensor, _input_ids: Optional[torch.Tensor]) -> torch.Tensor:
            # logits shape: (batch, vocab)
            if logits.ndim != 2:
                # Support passing last-token logits; ensure 2D
                logits_ = logits.view(logits.size(0), -1)
            else:
                logits_ = logits
            scores = logits_.clone()
            # top-k
            if top_k is not None and top_k > 0:
                kth_vals, _ = torch.topk(scores, k=min(top_k, scores.size(-1)), dim=-1)
                kth = kth_vals[:, -1].unsqueeze(-1)
                mask = scores < kth
                scores[mask] = float("-inf")
            # top-p (nucleus)
            if top_p is not None and 0.0 < top_p < 1.0:
                probs = torch.softmax(scores, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumprobs > top_p
                # Ensure at least one token kept
                cutoff[:, 0] = False
                # Map back to original indices
                to_remove = torch.zeros_like(scores, dtype=torch.bool)
                to_remove.scatter_(1, sorted_idx, cutoff)
                scores[to_remove] = float("-inf")
            return scores

        return processor

    def _set_logits_processor(
        self, processor: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]]
    ) -> None:
        try:
            sm = getattr(self.engine, "model_runner").sampling_manager  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"Failed to access sampling manager: {e}")
            return
        if processor is None:
            # Set a pass-through
            sm.set_logits_processor(lambda l, _i: l)
        else:
            sm.set_logits_processor(processor)

    # -----------------------------
    # Factual analysis helpers
    # -----------------------------
    def _append_references_section(self, text: str, references: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        lines.append("\n\nReferences:")
        for i, ref in enumerate(references, start=1):
            name = ref.get("name") or ref.get("text") or "Reference"
            uri = ref.get("uri") or ref.get("url") or ref.get("link") or ""
            ont = ref.get("ontology") or ""
            code = ref.get("code") or ""
            suffix = f" ({ont}:{code})" if ont or code else ""
            if uri:
                lines.append(f"[{i}] {name}{suffix} - {uri}")
            else:
                lines.append(f"[{i}] {name}{suffix}")
        return text + "\n" + "\n".join(lines)

    def _analyze_factual_consistency(self, text: str) -> Dict[str, Any]:
        """Lightweight factual analysis using NERProcessor and simple heuristics.
        Returns a report dict with:
          - references: List[{text, name, code, ontology, uri, score, citation_id}]
          - contradictions: List[{entity_text, code, ontology}]
          - speculative_sentences: List[{index, text}]
          - sentence_confidence: List[{index, text, score}]
          - text_with_citations: Optional[str] if inline citation insertion performed
        """
        try:
            from .ner_processor import NERProcessor  # lazy import
        except Exception:
            return {"enabled": False, "error": "ner_processor_unavailable"}

        # Extract and link entities
        try:
            proc = NERProcessor(inference_pipeline=None, config=None)
            ner = proc.extract_entities(text)
            linked = proc.link_entities(ner, ontology=self.constraints.ontology_default)
        except Exception as e:
            return {"enabled": True, "error": str(e)}

        ents = linked.entities if hasattr(linked, "entities") else []
        # Build references list and support inline citation IDs
        references: List[Dict[str, Any]] = []
        code_to_id: Dict[str, int] = {}
        occurrences: List[Tuple[int, int, int]] = []  # (start, end, citation_id)

        def _norm_key(s: str) -> str:
            s_ = s.strip().lower()
            import re as _re

            s_ = _re.sub(r"\s+", " ", s_)
            return s_

        for ent in ents:
            links = ent.get("ontology_links") or []
            if not links:
                continue
            link0 = links[0]
            code = str(link0.get("code") or "")
            if code and code not in code_to_id:
                references.append(
                    {
                        "text": ent.get("text"),
                        "type": ent.get("type"),
                        "code": code,
                        "ontology": link0.get("ontology"),
                        "name": link0.get("name"),
                        "score": link0.get("score"),
                        "uri": link0.get("uri"),
                    }
                )
                code_to_id[code] = len(references)  # 1-based
            # record first occurrence span per code for inline citation
            if code:
                cid = code_to_id[code]
                start = int(ent.get("start", -1))
                end = int(ent.get("end", -1))
                if start >= 0 and end >= start:
                    # only keep earliest occurrence per code
                    if not any(c == cid for _a, _b, c in occurrences):
                        occurrences.append((start, end, cid))

        # Detect simple contradictions: same normalized entity appears negated and affirmed
        contradictions: List[Dict[str, Any]] = []
        if getattr(self.constraints, "contradiction_detection_enabled", False):
            by_key: Dict[str, Dict[str, bool]] = {}
            for ent in ents:
                key = f"{_norm_key(ent.get('text',''))}|{str(ent.get('type','')).lower()}"
                if key not in by_key:
                    by_key[key] = {"pos": False, "neg": False}
                neg = bool(ent.get("negated"))
                if neg:
                    by_key[key]["neg"] = True
                else:
                    by_key[key]["pos"] = True
            for key, flags in by_key.items():
                if flags["pos"] and flags["neg"]:
                    txt, typ = key.split("|", 1)
                    # Prefer first linked code for citation context if available
                    code = None
                    onto = None
                    for ent in ents:
                        if (
                            _norm_key(ent.get("text", "")) == txt
                            and str(ent.get("type", "")) == typ
                        ):
                            links = ent.get("ontology_links") or []
                            if links:
                                code = links[0].get("code")
                                onto = links[0].get("ontology")
                                break
                    contradictions.append(
                        {"entity_text": txt, "type": typ, "code": code, "ontology": onto}
                    )

        # Speculative/hedging detection and confidence scoring
        import re as _re

        sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        hedges = {h.lower() for h in getattr(self.constraints, "hedging_markers", [])}
        abs_markers = {"always", "never", "guarantee", "cure", "miracle"}
        speculative_sentences: List[Dict[str, Any]] = []
        sentence_conf: List[Dict[str, Any]] = []
        if getattr(self.constraints, "speculative_flagging_enabled", False) or getattr(
            self.constraints, "confidence_scoring_enabled", False
        ):
            for idx, s in enumerate(sentences):
                low = s.lower()
                is_spec = any(f" {w} " in f" {low} " for w in hedges)
                abs_hit = any(w in low for w in abs_markers)
                score = 0.6
                # boost if contains a high-confidence link
                has_high_link = False
                for ent in ents:
                    if int(ent.get("start", -1)) < 0:
                        continue
                    s_text = text[int(ent.get("start")) : int(ent.get("end", 0))]
                    if s_text and s_text in s:
                        links = ent.get("ontology_links") or []
                        if links and float(links[0].get("score", 0.0)) >= 0.8:
                            has_high_link = True
                            break
                if has_high_link:
                    score += 0.2
                if is_spec:
                    score -= 0.2
                if abs_hit:
                    score -= 0.2
                score = max(0.0, min(1.0, score))
                if getattr(self.constraints, "speculative_flagging_enabled", False) and is_spec:
                    speculative_sentences.append({"index": idx, "text": s})
                if getattr(self.constraints, "confidence_scoring_enabled", False):
                    sentence_conf.append({"index": idx, "text": s, "score": score})

        # Inline citations insertion (optional in caller)
        text_with_citations = None
        if getattr(self.constraints, "add_inline_citations", False) and occurrences:
            # Build once by inserting markers from left to right
            occurrences.sort(key=lambda x: x[0])
            out: List[str] = []
            cur = 0
            for start, end, cid in occurrences:
                if start < cur:
                    continue
                out.append(text[cur:start])
                out.append(text[start:end])
                out.append(f"[{cid}]")
                cur = end
            out.append(text[cur:])
            text_with_citations = "".join(out)

        report: Dict[str, Any] = {
            "enabled": True,
            "references": references,
            "contradictions": contradictions,
            "speculative_sentences": speculative_sentences,
            "sentence_confidence": sentence_conf,
        }
        if text_with_citations is not None:
            report["text_with_citations"] = text_with_citations
        return report

    # -----------------------------
    # Style preset I/O helpers
    # -----------------------------
    def save_style_preset(self, path: str, preset: Dict[str, Any]) -> None:
        """Save a style preset dict to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2, ensure_ascii=False)

    def load_style_preset(self, path: str) -> Dict[str, Any]:
        """Load a style preset dict from a JSON file. Returns empty dict on failure."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Failed to load style preset '{path}': {e}")
        return {}
