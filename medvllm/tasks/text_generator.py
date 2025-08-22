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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import logging

import torch

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

        styled_prompt = self._apply_style(prompt, style)
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

        # Finally, apply generic content filters (banned phrases, disclaimer, etc.)
        filtered = self.constraints.apply_output_filters(formatted_text)

        meta: Dict[str, Any] = {
            "strategy": strategy,
            "temperature": effective_temperature,
            "top_p": top_p,
            "top_k": top_k,
            "beam_width": beam_width,
            "style": style,
            "purpose": purpose,
            "constraints_report": constraints_report,
        }

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
        if not style:
            return prompt
        style = style.lower()
        if style == "concise":
            prefix = "Please respond concisely and to the point.\n\n"
        elif style == "formal":
            prefix = "Use a formal, clinical tone suitable for medical documentation.\n\n"
        elif style == "patient_friendly":
            prefix = "Explain in plain language suitable for patients. Avoid jargon.\n\n"
        else:
            prefix = f"Style: {style}. Write accordingly.\n\n"
        return prefix + prompt

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
