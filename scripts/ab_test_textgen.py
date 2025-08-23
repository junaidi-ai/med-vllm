#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any, Dict, List

from medvllm.tasks import TextGenerator, MedicalConstraints
from medvllm.metrics import corpus_bleu, corpus_rouge_l, train_unigram_lm, perplexity


class EchoEngine:
    """A minimal engine-like object for offline A/B comparisons.
    It echoes the prompt with a simple transformation based on strategy/temperature.
    """

    def __init__(self) -> None:
        self.counter = 0

    def generate(self, prompts: List[str], sampling_params: Any, use_tqdm: bool = False) -> List[dict]:
        prompt = prompts[0]
        self.counter += 1
        # Simulate variability via counter and temperature
        temp = getattr(sampling_params, "temperature", 0.0)
        tag = "GREEDY" if temp == 0.0 else f"SAMPLED{self.counter%3}"
        text = f"{tag}: {prompt}"
        return [{"text": text, "prompt": prompt}]


def load_dataset(path: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def run_ab(
    dataset_path: str,
    output_path: str,
    strategies: List[str],
    temperature: float = 0.7,
    top_p: float | None = 0.9,
    beam_width: int = 3,
    use_echo_engine: bool = True,
) -> None:
    data = load_dataset(dataset_path)
    constraints = MedicalConstraints(enforce_disclaimer=False, ontology_verify_enabled=False)

    engine = EchoEngine() if use_echo_engine else None
    tg = TextGenerator(engine if engine else "gpt2", constraints=constraints)

    # Build reference lists for metrics (per-item)
    references = [d["reference"] for d in data]
    lm = train_unigram_lm(references)

    results: Dict[str, Any] = {
        "dataset": dataset_path,
        "strategies": strategies,
        "created_at": int(time.time()),
        "items": [],
        "summary": {},
    }

    for strat in strategies:
        gens: List[str] = []
        per_item: List[Dict[str, Any]] = []
        for d in data:
            res = tg.generate(
                d["prompt"],
                strategy=strat,
                temperature=temperature,
                top_p=top_p,
                beam_width=beam_width,
                max_length=256,
                readability="general",
                tone="formal",
            )
            gens.append(res.generated_text)
            per_item.append({
                "id": d.get("id"),
                "prompt": d.get("prompt"),
                "reference": d.get("reference"),
                "generated_text": res.generated_text,
                "metadata": res.metadata,
            })

        bleu = corpus_bleu(references, gens)
        rouge = corpus_rouge_l(references, gens)
        ppl = sum(perplexity(g, lm) for g in gens) / max(1, len(gens))

        results["items"].append({
            "strategy": strat,
            "records": per_item,
            "metrics": {
                "corpus_bleu": bleu,
                "corpus_rouge_l": rouge,
                "avg_unigram_perplexity": ppl,
            }
        })

    # Summarize best by each metric
    summary: Dict[str, Any] = {}
    for metric in ["corpus_bleu", "corpus_rouge_l", "avg_unigram_perplexity"]:
        best = None
        for strat_block in results["items"]:
            m = strat_block["metrics"][metric]
            if best is None:
                best = (strat_block["strategy"], m)
            else:
                if metric == "avg_unigram_perplexity":  # lower is better
                    if m < best[1]:
                        best = (strat_block["strategy"], m)
                else:  # higher is better
                    if m > best[1]:
                        best = (strat_block["strategy"], m)
        summary[metric] = {"best_strategy": best[0], "score": best[1] if best else None}
    results["summary"] = summary

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote A/B results to {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B test TextGenerator strategies")
    ap.add_argument("--dataset", default="benchmarks/datasets/textgen_small.jsonl")
    ap.add_argument("--output", default="benchmark_results_cpu_smoke/textgen_ab_results.json")
    ap.add_argument("--strategies", nargs="*", default=["greedy", "sampling", "beam", "template"]) 
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--beam_width", type=int, default=3)
    ap.add_argument("--no-echo", action="store_true", help="Use real engine via model name (env MEDVLLM_MODEL)")
    args = ap.parse_args()

    use_echo = not args.no_echo

    run_ab(
        dataset_path=args.dataset,
        output_path=args.output,
        strategies=list(args.strategies),
        temperature=args.temperature,
        top_p=args.top_p,
        beam_width=args.beam_width,
        use_echo_engine=use_echo,
    )


if __name__ == "__main__":
    main()
