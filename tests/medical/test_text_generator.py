from __future__ import annotations

from typing import List

from medvllm.tasks import TextGenerator, MedicalConstraints
from medvllm.sampling_params import SamplingParams


class FakeEngine:
    """Engine-like object for testing TextGenerator.

    - Greedy (temperature=0) returns a fixed string.
    - Sampling (temperature>0) returns varying strings across calls to simulate non-determinism.
    """

    def __init__(self) -> None:
        self.counter = 0

    def generate(
        self, prompts: List[str], sampling_params: SamplingParams, use_tqdm: bool = False
    ) -> List[dict]:
        prompt = prompts[0]
        if sampling_params.temperature == 0.0:
            text = f"GREEDY RESPONSE -> {prompt}"
        else:
            self.counter += 1
            text = (
                f"SAMPLED RESPONSE #{self.counter} (temp={sampling_params.temperature}) -> {prompt}"
            )
        return [{"text": text, "prompt": prompt}]


def test_greedy_vs_sampling_determinism() -> None:
    tg = TextGenerator(FakeEngine(), constraints=MedicalConstraints(enforce_disclaimer=False))
    prompt = "Explain the management of type 2 diabetes mellitus."

    # Greedy should be deterministic across runs
    g1 = tg.generate(prompt, strategy="greedy", max_length=32)
    g2 = tg.generate(prompt, strategy="greedy", max_length=32)
    assert g1.generated_text == g2.generated_text
    assert "GREEDY RESPONSE" in g1.generated_text

    # Sampling should differ across consecutive runs (simulated non-determinism)
    s1 = tg.generate(prompt, strategy="sampling", temperature=0.8, top_p=0.9, max_length=32)
    s2 = tg.generate(prompt, strategy="sampling", temperature=0.8, top_p=0.9, max_length=32)
    assert s1.generated_text != s2.generated_text
    assert "SAMPLED RESPONSE" in s1.generated_text


def test_banned_phrase_redaction_and_disclaimer() -> None:
    constraints = MedicalConstraints(
        banned_phrases=["foobar"],
        required_disclaimer="Disclaimer: Not medical advice.",
        enforce_disclaimer=True,
        max_length_chars=None,
    )
    tg = TextGenerator(FakeEngine(), constraints=constraints)

    prompt = "Provide a brief overview of hypertension management. Avoid foobar in answers."
    res = tg.generate(prompt, strategy="greedy", max_length=64)

    assert "[REDACTED]" in res.generated_text
    assert "Disclaimer: Not medical advice." in res.generated_text


def test_generate_with_context_prefix() -> None:
    tg = TextGenerator(FakeEngine(), constraints=MedicalConstraints(enforce_disclaimer=False))
    prompt = "Summarize treatment options."
    context = "Adult patient with stage 1 hypertension and diabetes."

    res = tg.generate_with_context(prompt=prompt, context=context, strategy="greedy", max_length=48)
    # The engine echoes a slice of the prompt part; ensure context prefix is included in the echoed text
    assert "Context:" in res.generated_text
    assert "Prompt:" in res.generated_text
