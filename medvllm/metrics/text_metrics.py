from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import math
from collections import Counter


def tokenize(text: str) -> List[str]:
    # Very simple whitespace tokenizer; lowercase for robustness
    return [t for t in text.strip().lower().split() if t]


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def bleu_score(reference: str, candidate: str, max_n: int = 4, smooth: float = 1.0) -> float:
    """Compute a simple BLEU with uniform n-gram weights and add-k smoothing.
    - reference, candidate: raw strings
    - max_n: up to which n-gram to include (default 4)
    - smooth: add-k smoothing value (default 1.0)
    Returns BLEU in [0,1].
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not cand_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = _ngrams(ref_tokens, n)
        cand_ngrams = _ngrams(cand_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        overlap = 0
        for ng, c in cand_ngrams.items():
            overlap += min(c, ref_ngrams.get(ng, 0))
        p_n = (overlap + smooth) / (sum(cand_ngrams.values()) + smooth)
        precisions.append(p_n)

    # geometric mean of precisions
    log_prec = sum(math.log(p) for p in precisions if p > 0) / max(
        1, len([p for p in precisions if p > 0])
    )

    # brevity penalty
    r = len(ref_tokens)
    c = len(cand_tokens)
    if c == 0:
        return 0.0
    bp = 1.0 if c > r else math.exp(1 - float(r) / max(1, c))

    return max(0.0, min(1.0, bp * math.exp(log_prec)))


def corpus_bleu(
    references: Sequence[str], candidates: Sequence[str], max_n: int = 4, smooth: float = 1.0
) -> float:
    if not references or not candidates or len(references) != len(candidates):
        return 0.0
    # aggregate n-gram counts
    ref_counts = [Counter() for _ in range(max_n)]
    cand_counts = [Counter() for _ in range(max_n)]

    r_total = 0
    c_total = 0

    for ref, cand in zip(references, candidates):
        rt = tokenize(ref)
        ct = tokenize(cand)
        r_total += len(rt)
        c_total += len(ct)
        for n in range(1, max_n + 1):
            ref_counts[n - 1] += _ngrams(rt, n)
            cand_counts[n - 1] += _ngrams(ct, n)

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        overlap = 0
        for ng, c in cand_counts[n - 1].items():
            overlap += min(c, ref_counts[n - 1].get(ng, 0))
        denom = sum(cand_counts[n - 1].values())
        p_n = (overlap + smooth) / (denom + smooth) if denom > 0 else 0.0
        precisions.append(p_n)

    nonzero = [p for p in precisions if p > 0]
    log_prec = sum(math.log(p) for p in nonzero) / max(1, len(nonzero))
    bp = 1.0 if c_total > r_total else math.exp(1 - float(r_total) / max(1, c_total))
    return max(0.0, min(1.0, bp * math.exp(log_prec)))


def rouge_l(reference: str, candidate: str) -> float:
    """Compute ROUGE-L F-measure using LCS between reference and candidate tokens.
    Returns a value in [0,1].
    """
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0

    # DP for LCS length
    m, n = len(ref), len(cand)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == cand[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    prec = lcs / max(1, n)
    rec = lcs / max(1, m)
    if prec + rec == 0:
        return 0.0
    beta2 = 1.2 * 1.2  # typical for ROUGE-L
    f = (1 + beta2) * prec * rec / (rec + beta2 * prec)
    return max(0.0, min(1.0, f))


def corpus_rouge_l(references: Sequence[str], candidates: Sequence[str]) -> float:
    if not references or not candidates or len(references) != len(candidates):
        return 0.0
    scores = [rouge_l(r, c) for r, c in zip(references, candidates)]
    return sum(scores) / len(scores) if scores else 0.0


# Very simple unigram language model for approximate perplexity
class UnigramLM:
    def __init__(self, counts: Counter, total: int, vocab_size: int, alpha: float = 1.0) -> None:
        self.counts = counts
        self.total = total
        self.vocab_size = vocab_size
        self.alpha = alpha

    def prob(self, token: str) -> float:
        # Additive smoothing
        return (self.counts.get(token, 0) + self.alpha) / (
            self.total + self.alpha * self.vocab_size
        )


def train_unigram_lm(references: Sequence[str], alpha: float = 1.0) -> UnigramLM:
    tokens: List[str] = []
    for r in references:
        tokens.extend(tokenize(r))
    counts = Counter(tokens)
    vocab = len(counts) if counts else 1
    total = sum(counts.values())
    return UnigramLM(counts, total, vocab, alpha=alpha)


def perplexity(text: str, lm: UnigramLM) -> float:
    tokens = tokenize(text)
    if not tokens:
        return float('inf')
    logp = 0.0
    for t in tokens:
        p = max(1e-12, lm.prob(t))
        logp += math.log(p)
    avg_logp = logp / len(tokens)
    # Perplexity = exp(-avg_log_prob)
    return math.exp(-avg_logp)


# ------------------------------
# ROUGE-1/2 and corpus precision/recall
# ------------------------------
def _ngram_overlap_stats(
    ref_tokens: Sequence[str], cand_tokens: Sequence[str], n: int
) -> Tuple[int, int, int]:
    """Return (overlap, cand_total, ref_total) for n-grams of order n."""
    ref_ngr = _ngrams(ref_tokens, n)
    cand_ngr = _ngrams(cand_tokens, n)
    overlap = 0
    for ng, c in cand_ngr.items():
        overlap += min(c, ref_ngr.get(ng, 0))
    return overlap, sum(cand_ngr.values()), sum(ref_ngr.values())


def rouge_n(ref: str, cand: str, n: int = 1) -> Tuple[float, float, float]:
    """Compute ROUGE-N (precision, recall, F1) on token n-grams.
    Returns a tuple (precision, recall, f1) each in [0,1].
    """
    rt = tokenize(ref)
    ct = tokenize(cand)
    if not rt or not ct or n <= 0:
        return 0.0, 0.0, 0.0
    overlap, cand_total, ref_total = _ngram_overlap_stats(rt, ct, n)
    prec = overlap / cand_total if cand_total > 0 else 0.0
    rec = overlap / ref_total if ref_total > 0 else 0.0
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def corpus_rouge_n(
    refs: Sequence[str], cands: Sequence[str], n: int = 1, average: str = "micro"
) -> Tuple[float, float, float]:
    """Corpus ROUGE-N with either micro (sum-overlaps) or macro (avg per item) averaging.
    Returns (precision, recall, f1).
    """
    if not refs or not cands or len(refs) != len(cands):
        return 0.0, 0.0, 0.0
    if average not in {"micro", "macro"}:
        average = "micro"

    if average == "micro":
        overlap_sum = 0
        cand_sum = 0
        ref_sum = 0
        for r, c in zip(refs, cands):
            o, ctot, rtot = _ngram_overlap_stats(tokenize(r), tokenize(c), n)
            overlap_sum += o
            cand_sum += ctot
            ref_sum += rtot
        prec = overlap_sum / cand_sum if cand_sum > 0 else 0.0
        rec = overlap_sum / ref_sum if ref_sum > 0 else 0.0
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        return prec, rec, f1
    else:
        ps: List[float] = []
        rs: List[float] = []
        fs: List[float] = []
        for r, c in zip(refs, cands):
            p, r_, f = rouge_n(r, c, n=n)
            ps.append(p)
            rs.append(r_)
            fs.append(f)
        prec = sum(ps) / len(ps)
        rec = sum(rs) / len(rs)
        f1 = sum(fs) / len(fs)
        return prec, rec, f1
