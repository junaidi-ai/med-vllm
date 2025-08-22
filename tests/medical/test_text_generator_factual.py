from __future__ import annotations

from typing import List

from medvllm.tasks import TextGenerator, MedicalConstraints
from tests.medical.test_text_generator import FakeEngine


def _make_tg(**constraint_overrides: object) -> TextGenerator:
    base = {
        "enforce_disclaimer": False,
        # keep defaults for factual toggles unless overridden
    }
    base.update(constraint_overrides)
    constraints = MedicalConstraints(**base)  # type: ignore[arg-type]
    return TextGenerator(FakeEngine(), constraints=constraints)


def test_inline_citations_and_references_insertion() -> None:
    tg = _make_tg(
        factual_checks_enabled=True,
        add_inline_citations=True,
        append_references_section=True,
    )
    # Include 2 distinct, linkable entities in order: pneumonia, aspirin
    prompt = "Pneumonia may be treated with supportive care; aspirin is for pain and fever control."

    res = tg.generate(prompt, strategy="greedy", max_length=256)

    # Inline citation markers should be inserted in order of first occurrence
    assert "[1]" in res.generated_text
    assert "[2]" in res.generated_text
    # References section appended
    assert "References:" in res.generated_text
    # Expect UMLS CUIs and URIs for pneumonia (C0032310) and aspirin (C0004057)
    assert "(UMLS:C0032310)" in res.generated_text
    assert "https://uts.nlm.nih.gov/umls/concept/C0032310" in res.generated_text
    assert "(UMLS:C0004057)" in res.generated_text
    assert "https://uts.nlm.nih.gov/umls/concept/C0004057" in res.generated_text

    # Metadata factual report should include two references
    fr = res.metadata.get("factual_report", {})
    refs = fr.get("references", [])
    codes = {r.get("code") for r in refs}
    assert {"C0032310", "C0004057"}.issubset(codes)


def test_contradiction_detection_negated_and_affirmed() -> None:
    tg = _make_tg(
        factual_checks_enabled=True,
        contradiction_detection_enabled=True,
        add_inline_citations=False,
        append_references_section=False,
    )
    # First mention negated, second affirmed of same entity ("pneumonia")
    prompt = "The patient denies pneumonia. Later, pneumonia was documented after imaging."

    res = tg.generate(prompt, strategy="greedy", max_length=256)
    fr = res.metadata.get("factual_report", {})
    contras = fr.get("contradictions", [])

    assert isinstance(contras, list)
    assert len(contras) >= 1
    # Check that pneumonia is flagged
    assert any("pneumonia" in (c.get("entity_text", "")) for c in contras)
    # Optional: linked code should be present for context
    c_with_code = next((c for c in contras if c.get("code")), None)
    assert c_with_code is not None
    assert c_with_code.get("code") in {"C0032310"}


def test_speculative_flagging_and_confidence_scoring() -> None:
    tg = _make_tg(
        factual_checks_enabled=True,
        speculative_flagging_enabled=True,
        confidence_scoring_enabled=True,
        add_inline_citations=False,
        append_references_section=False,
    )
    # Sentences:
    #  0: GREEDY prefix + first sentence; keep a non-med sentence to avoid link boost
    #  1: Hedged, no linkable entity -> speculative; score 0.4 (0.6 - 0.2)
    #  2: Contains high-confidence linked entity (pneumonia), no hedging -> score 0.8 (0.6 + 0.2)
    prompt = "This is a general statement. It may be viral illness. Pneumonia is present."

    res = tg.generate(prompt, strategy="greedy", max_length=256)
    fr = res.metadata.get("factual_report", {})

    specs = fr.get("speculative_sentences", [])
    assert any("may" in s.get("text", "").lower() for s in specs)

    confs = fr.get("sentence_confidence", [])
    # Find the sentence with pneumonia
    pneu = next((s for s in confs if "pneumonia is present" in s.get("text", "").lower()), None)
    assert pneu is not None
    assert abs(float(pneu.get("score", 0.0)) - 0.8) < 1e-6

    # Find the speculative viral illness sentence
    viral = next((s for s in confs if "viral illness" in s.get("text", "").lower()), None)
    assert viral is not None
    assert abs(float(viral.get("score", 0.0)) - 0.4) < 1e-6


def test_confidence_boost_high_conf_linked_entity() -> None:
    tg = _make_tg(
        factual_checks_enabled=True,
        confidence_scoring_enabled=True,
        speculative_flagging_enabled=False,
    )
    # First sentence has a high-confidence entity; second is plain text.
    prompt = "Pneumonia is suspected. The plan is to follow up."

    res = tg.generate(prompt, strategy="greedy", max_length=256)
    confs = res.metadata.get("factual_report", {}).get("sentence_confidence", [])

    # pneumonia sentence should have >= 0.8; the other near default 0.6
    pneu = next((s for s in confs if "pneumonia is suspected" in s.get("text", "").lower()), None)
    follow = next((s for s in confs if "plan is to follow up" in s.get("text", "").lower()), None)
    assert pneu is not None and follow is not None
    assert float(pneu.get("score", 0.0)) >= 0.8
    assert abs(float(follow.get("score", 0.0)) - 0.6) < 1e-6
