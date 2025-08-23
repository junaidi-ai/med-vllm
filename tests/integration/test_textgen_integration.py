from __future__ import annotations

import json
from pathlib import Path

from medvllm.tasks import TextGenerator, MedicalConstraints
from medvllm.metrics import corpus_bleu, corpus_rouge_l, train_unigram_lm, perplexity
from tests.medical.test_text_generator import FakeEngine


def test_textgen_pipeline_metrics(tmp_path: Path) -> None:
    # Load small dataset
    ds_path = Path("benchmarks/datasets/textgen_small.jsonl")
    items = [json.loads(l) for l in ds_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    tg = TextGenerator(FakeEngine(), constraints=MedicalConstraints(enforce_disclaimer=False))

    refs = [it["reference"] for it in items]
    lm = train_unigram_lm(refs)

    gens = []
    for it in items:
        res = tg.generate(it["prompt"], strategy="greedy", max_length=128)
        gens.append(res.generated_text)

    # Metrics run and are within sane ranges
    bleu = corpus_bleu(refs, gens)
    rouge = corpus_rouge_l(refs, gens)
    ppl = sum(perplexity(g, lm) for g in gens) / max(1, len(gens))

    assert 0.0 <= bleu <= 1.0
    assert 0.0 <= rouge <= 1.0
    assert ppl > 0.0
