from __future__ import annotations

from medvllm.metrics import bleu_score, rouge_l, rouge_n, train_unigram_lm, perplexity


def test_bleu_basic_overlap() -> None:
    ref = "the cat is on the mat"
    cand = "the cat is on the mat"
    assert bleu_score(ref, cand) > 0.9


def test_bleu_partial_overlap_lower() -> None:
    ref = "the cat is on the mat"
    cand = "the cat on mat"
    assert 0.1 < bleu_score(ref, cand) < 1.0


def test_rouge_l_exact_and_partial() -> None:
    ref = "pneumonia symptoms include fever and cough"
    cand_exact = ref
    cand_partial = "fever and cough are common"
    assert rouge_l(ref, cand_exact) > 0.9
    assert 0.2 < rouge_l(ref, cand_partial) < 1.0


def test_unigram_perplexity_reasonable() -> None:
    refs = [
        "aspirin reduces pain and fever",
        "pneumonia may cause fever and cough",
    ]
    lm = train_unigram_lm(refs)
    text_in_domain = "fever and cough"
    text_oov = "quantum entanglement"
    ppl_in = perplexity(text_in_domain, lm)
    ppl_oov = perplexity(text_oov, lm)
    assert ppl_in < ppl_oov  # in-domain should be less perplexing


def test_rouge_n_unigram_and_bigram() -> None:
    ref = "the quick brown fox jumps over the lazy dog"
    cand = "the quick fox jumps over dog"
    p1, r1, f1_1 = rouge_n(ref, cand, n=1)
    p2, r2, f1_2 = rouge_n(ref, cand, n=2)
    # ROUGE-1 should be reasonably high; ROUGE-2 lower due to missing bigrams
    assert 0.5 < f1_1 <= 1.0
    assert 0.0 <= f1_2 < f1_1
