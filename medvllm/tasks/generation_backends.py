from __future__ import annotations

from typing import Protocol


class BackendAdapter(Protocol):
    def prepare_prompt(self, prompt: str) -> str:  # pragma: no cover - simple glue
        ...

    def postprocess(self, text: str) -> str:  # pragma: no cover - simple glue
        ...


class GPTAdapter:
    def prepare_prompt(self, prompt: str) -> str:
        return prompt

    def postprocess(self, text: str) -> str:
        return text


class T5Adapter:
    def prepare_prompt(self, prompt: str) -> str:
        # Minimal T5-style instruction prefixing.
        # Real T5 tasks often look like: "summarize: ...", "translate English to German: ...".
        # Keep it generic to avoid constraining behavior.
        if any(prefix in prompt.lower() for prefix in ("summarize:", "translate ", "classify:")):
            return prompt
        return f"instruction: {prompt}"

    def postprocess(self, text: str) -> str:
        return text


class BERTAdapter:
    def prepare_prompt(self, prompt: str) -> str:
        # BERT is not a generative decoder by default; for a simple integration,
        # we ask the underlying (decoder) engine to produce a sentence that fits.
        return f"Fill in the blanks and produce a fluent sentence: {prompt}"

    def postprocess(self, text: str) -> str:
        return text


def create_backend(name: str | None) -> BackendAdapter:
    n = (name or "gpt").lower()
    if n in ("gpt", "auto", "llm"):
        return GPTAdapter()
    if n in ("t5",):
        return T5Adapter()
    if n in ("bert",):
        return BERTAdapter()
    # Default to pass-through
    return GPTAdapter()
