"""Example usage of TextGenerator.

This example uses a lightweight FakeEngine for demonstration. To use a real
model, replace FakeEngine with `LLM("your-model-id")`.
"""

from __future__ import annotations

from typing import Any, List

from medvllm.tasks import TextGenerator, MedicalConstraints
from medvllm.sampling_params import SamplingParams


class FakeEngine:
    """Minimal engine-like object compatible with TextGenerator.

    It implements .generate(prompts, sampling_params, use_tqdm=False) and returns
    a list of dicts with a 'text' key.
    """

    def generate(
        self, prompts: List[str], sampling_params: SamplingParams, use_tqdm: bool = False
    ) -> List[dict]:
        prompt = prompts[0]
        # Show difference by temperature
        if sampling_params.temperature == 0.0:
            text = f"GREEDY RESPONSE -> {prompt[:60]}"
        else:
            text = f"SAMPLED RESPONSE (temp={sampling_params.temperature}) -> {prompt[:60]}"
        return [{"text": text, "prompt": prompt}]


def main() -> None:
    constraints = MedicalConstraints(
        banned_phrases=["foobar"],
        required_disclaimer="Disclaimer: Not medical advice.",
        enforce_disclaimer=True,
        max_length_chars=400,
    )

    tg = TextGenerator(FakeEngine(), constraints=constraints)

    prompt = "Provide a brief overview of hypertension management. Avoid foobar."

    res_greedy = tg.generate(prompt, strategy="greedy", max_length=64)
    res_sampling = tg.generate(
        prompt, strategy="sampling", temperature=0.8, top_p=0.9, max_length=64
    )
    res_context = tg.generate_with_context(
        prompt="Summarize treatment options.",
        context="Adult patient with stage 1 hypertension and diabetes.",
        strategy="beam",
        beam_width=3,
        max_length=64,
        style="patient_friendly",
    )

    print("\n--- Greedy ---\n", res_greedy.generated_text)
    print("\n--- Sampling ---\n", res_sampling.generated_text)
    print("\n--- With Context ---\n", res_context.generated_text)


if __name__ == "__main__":
    main()
