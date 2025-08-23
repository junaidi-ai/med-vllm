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

    # New: Template-based strategy
    res_template = tg.generate(
        "Explain what HbA1c measures.",
        strategy="template",
        template="Instruction: {prompt}\nOutput: Provide a lay explanation.",
        max_length=64,
    )

    # New: Few-shot strategy
    few_shot_examples = [
        {"input": "What is hypertension?", "output": "High blood pressure."},
        ("What is BMI?", "Body mass index."),
    ]
    res_few_shot = tg.generate(
        "Define tachycardia.",
        strategy="few_shot",
        few_shot_examples=few_shot_examples,
        temperature=0.7,
        max_length=96,
    )

    # New: Backend selection (T5-style prompt preparation)
    res_t5_backend = tg.generate(
        "Summarize the patient case.",
        strategy="greedy",
        backend="t5",
        max_length=64,
    )

    # New: Fine-grained length and style controls
    res_controls = tg.generate(
        "Create discharge instructions for pneumonia.",
        strategy="beam",
        beam_width=3,
        max_length=128,
        readability="general",  # audience: general public
        tone="friendly",  # conversational tone
        structure="bullet",  # bullet points
        specialty="pulmonology",  # domain focus
        target_words=80,  # soft cap
        target_chars=600,  # hard cap via constraints
    )

    # New: Load and apply a style preset (JSON)
    # See: examples/presets/patient_education.json
    res_preset = tg.generate(
        "Patient education on diabetes foot care.",
        strategy="greedy",
        style_preset="examples/presets/patient_education.json",
        max_length=96,
    )

    print("\n--- Greedy ---\n", res_greedy.generated_text)
    print("\n--- Sampling ---\n", res_sampling.generated_text)
    print("\n--- With Context ---\n", res_context.generated_text)
    print("\n--- Template Strategy ---\n", res_template.generated_text)
    print("\n--- Few-shot Strategy ---\n", res_few_shot.generated_text)
    print("\n--- T5 Backend ---\n", res_t5_backend.generated_text)
    print("\n--- Style Controls ---\n", res_controls.generated_text)
    print("\n--- Preset (patient education) ---\n", res_preset.generated_text)


if __name__ == "__main__":
    main()
