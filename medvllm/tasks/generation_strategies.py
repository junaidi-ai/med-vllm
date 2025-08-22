from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

from medvllm.sampling_params import SamplingParams


Runner = Callable[[str, SamplingParams], str]


class BaseStrategy:
    def generate(
        self,
        prompt: str,
        run_once: Runner,
        *,
        max_length: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        beam_width: int,
        ignore_eos: bool,
    ) -> str:
        raise NotImplementedError


class GreedyStrategy(BaseStrategy):
    def generate(
        self,
        prompt: str,
        run_once: Runner,
        *,
        max_length: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        beam_width: int,
        ignore_eos: bool,
    ) -> str:
        sp = SamplingParams(temperature=0.0, max_tokens=max_length, ignore_eos=ignore_eos)
        return run_once(prompt, sp)


class SamplingStrategy(BaseStrategy):
    def generate(
        self,
        prompt: str,
        run_once: Runner,
        *,
        max_length: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        beam_width: int,
        ignore_eos: bool,
    ) -> str:
        sp = SamplingParams(temperature=temperature, max_tokens=max_length, ignore_eos=ignore_eos)
        return run_once(prompt, sp)


class SimpleBeamStrategy(BaseStrategy):
    def generate(
        self,
        prompt: str,
        run_once: Runner,
        *,
        max_length: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        beam_width: int,
        ignore_eos: bool,
    ) -> str:
        candidates: List[str] = []
        width = max(1, beam_width)
        for i in range(width):
            sp = SamplingParams(
                temperature=temperature if i > 0 else 0.0,
                max_tokens=max_length,
                ignore_eos=ignore_eos,
            )
            candidates.append(run_once(prompt, sp))
        # prefer first non-empty else longest
        for c in candidates:
            if c.strip():
                return c
        return max(candidates, key=len) if candidates else ""


@dataclass
class TemplateParams:
    template: str
    template_vars: Optional[dict] = None


class TemplateStrategy(BaseStrategy):
    def __init__(self, params: TemplateParams) -> None:
        self.params = params

    def generate(
        self,
        prompt: str,
        run_once: Runner,
        *,
        max_length: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        beam_width: int,
        ignore_eos: bool,
    ) -> str:
        vars_ = dict(self.params.template_vars or {})
        vars_.setdefault("prompt", prompt)
        try:
            return self.params.template.format(**vars_)
        except Exception:
            # Fall back to simple concatenation if formatting fails
            return f"{self.params.template}\n\n{prompt}"


@dataclass
class FewShotParams:
    examples: List[Any]
    example_separator: str = "\n\n"


class FewShotStrategy(BaseStrategy):
    def __init__(self, params: FewShotParams) -> None:
        self.params = params

    def _render_examples(self) -> str:
        rendered: List[str] = []
        for ex in self.params.examples:
            if isinstance(ex, str):
                rendered.append(ex)
            elif isinstance(ex, (list, tuple)) and len(ex) == 2:
                rendered.append(f"Input: {ex[0]}\nOutput: {ex[1]}")
            elif isinstance(ex, dict):
                inp = ex.get("input") or ex.get("prompt") or ""
                out = ex.get("output") or ex.get("completion") or ""
                rendered.append(f"Input: {inp}\nOutput: {out}")
            else:
                rendered.append(str(ex))
        return self.params.example_separator.join(rendered)

    def generate(
        self,
        prompt: str,
        run_once: Runner,
        *,
        max_length: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        beam_width: int,
        ignore_eos: bool,
    ) -> str:
        examples_block = self._render_examples()
        fs_prompt = (
            "You are a helpful medical assistant.\n\n"
            "Here are examples:\n\n"
            f"{examples_block}\n\n"
            "Now respond to the following.\n"
            f"Input: {prompt}\nOutput:"
        )
        sp = SamplingParams(
            temperature=max(temperature, 0.0), max_tokens=max_length, ignore_eos=ignore_eos
        )
        return run_once(fs_prompt, sp)


def create_strategy(
    name: str,
    *,
    template: Optional[str] = None,
    template_vars: Optional[dict] = None,
    few_shot_examples: Optional[List[Any]] = None,
) -> BaseStrategy:
    n = (name or "").lower()
    if n in ("greedy",):
        return GreedyStrategy()
    if n in ("sampling",):
        return SamplingStrategy()
    if n in ("beam", "simple_beam", "multi_sample"):
        return SimpleBeamStrategy()
    if n in ("template", "template_based"):
        tpl = template or "Instruction: {prompt}\nOutput:"
        return TemplateStrategy(TemplateParams(template=tpl, template_vars=template_vars))
    if n in ("few_shot", "fewshot"):
        return FewShotStrategy(FewShotParams(examples=few_shot_examples or []))
    raise ValueError(f"Unknown strategy: {name}")
