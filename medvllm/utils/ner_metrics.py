from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

Entity = Dict[str, object]
DocEntities = Sequence[Entity]


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    typ: str

    @staticmethod
    def from_entity(e: Entity) -> "_Span":
        return _Span(int(e["start"]), int(e["end"]), str(e["type"]).lower())

    def iou(self, other: "_Span") -> float:
        # Compute intersection-over-union on character spans [start, end)
        inter = max(0, min(self.end, other.end) - max(self.start, other.start))
        if inter == 0:
            return 0.0
        union = (self.end - self.start) + (other.end - other.start) - inter
        if union <= 0:
            return 0.0
        return inter / union


def _to_docs(entities: Iterable) -> List[DocEntities]:
    # Accept either [entities...] or [[entities...], ...]
    # Materialize to avoid consuming from a generator when peeking first element
    ent_list = list(entities)
    if not ent_list:
        return []
    first = ent_list[0]
    if isinstance(first, dict):
        return [list(ent_list)]  # type: ignore[list-item]
    return [list(doc) for doc in ent_list]  # type: ignore[list-item]


def _safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else float(n) / float(d)


def compute_ner_strict_metrics(
    y_true: Iterable, y_pred: Iterable, *, labels: Sequence[str] | None = None
) -> Dict[str, object]:
    """
    Compute span-and-type strict NER precision/recall/F1.

    Args:
        y_true: Iterable of entities or iterable of per-document entity lists.
                Each entity is a dict with at least {"start", "end", "type"}.
        y_pred: Same structure as y_true.
        labels: Optional list of entity types to include in per-type reporting.
                If None, inferred from gold types.

    Returns:
        dict with keys:
        - "micro": {"precision", "recall", "f1", "tp", "fp", "fn"}
        - "per_type": {type -> metrics dict as above}
    """
    gold_docs = _to_docs(y_true)
    pred_docs = _to_docs(y_pred)

    if len(gold_docs) != len(pred_docs):
        raise ValueError("y_true and y_pred must have the same number of documents")

    # Build sets for strict matching (exact span AND type)
    gold_sets: List[set[_Span]] = [{_Span.from_entity(e) for e in doc} for doc in gold_docs]
    pred_sets: List[set[_Span]] = [{_Span.from_entity(e) for e in doc} for doc in pred_docs]

    # Infer label set from gold if not provided
    if labels is None:
        label_set = sorted({sp.typ for doc in gold_sets for sp in doc})
    else:
        label_set = [lbl.lower() for lbl in labels]

    counts = _compute_ner_counts_and_metrics(gold_sets, pred_sets, label_set)
    # Backward-compatible shape: only micro and per_type
    return {"micro": counts["micro"], "per_type": counts["per_type"]}


def _compute_ner_counts_and_metrics(
    gold_sets: List[set[_Span]],
    pred_sets: List[set[_Span]],
    label_set: Sequence[str],
) -> Dict[str, object]:
    # Micro counts
    tp = fp = fn = 0

    # Per-type counts
    per_type_counts: Dict[str, Dict[str, int]] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in label_set}

    for gset, pset in zip(gold_sets, pred_sets):
        inter = gset & pset
        tp += len(inter)
        fp += len(pset - inter)
        fn += len(gset - inter)

        # Per-type strict
        for t in label_set:
            g_t = {s for s in gset if s.typ == t}
            p_t = {s for s in pset if s.typ == t}
            inter_t = g_t & p_t
            c = per_type_counts[t]
            c["tp"] += len(inter_t)
            c["fp"] += len(p_t - inter_t)
            c["fn"] += len(g_t - inter_t)

    micro_p = _safe_div(tp, tp + fp)
    micro_r = _safe_div(tp, tp + fn)
    micro_f1 = 0.0 if (micro_p + micro_r) == 0 else 2 * micro_p * micro_r / (micro_p + micro_r)

    per_type: Dict[str, Dict[str, float | int]] = {}
    for t, c in per_type_counts.items():
        p = _safe_div(c["tp"], c["tp"] + c["fp"])
        r = _safe_div(c["tp"], c["tp"] + c["fn"])
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        per_type[t] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
        }

    # Macro averages over types present in label_set
    macro_p = _safe_div(sum(pt["precision"] for pt in per_type.values()), len(per_type) or 1)
    macro_r = _safe_div(sum(pt["recall"] for pt in per_type.values()), len(per_type) or 1)
    macro_f1 = _safe_div(sum(pt["f1"] for pt in per_type.values()), len(per_type) or 1)

    return {
        "micro": {
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_type": per_type,
    }


def compute_ner_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    labels: Optional[Sequence[str]] = None,
    match: str = "strict",  # "strict" | "overlap"
    iou_threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Unified NER metrics with support for strict and overlap/IoU matching.

    Args:
        y_true, y_pred: Iterable of entities or iterable of per-document lists.
        labels: Optional entity types to include; inferred from gold if None.
        match: "strict" (exact span+type) or "overlap" (IoU >= threshold with same type).
        iou_threshold: IoU threshold for overlap matching.

    Returns:
        dict with keys: micro, macro, per_type.
    """
    gold_docs = _to_docs(y_true)
    pred_docs = _to_docs(y_pred)
    if len(gold_docs) != len(pred_docs):
        raise ValueError("y_true and y_pred must have the same number of documents")

    gold_lists: List[List[_Span]] = [[_Span.from_entity(e) for e in doc] for doc in gold_docs]
    pred_lists: List[List[_Span]] = [[_Span.from_entity(e) for e in doc] for doc in pred_docs]

    # Infer label set from gold if not provided
    if labels is None:
        label_set = sorted({sp.typ for doc in gold_lists for sp in doc})
    else:
        label_set = [lbl.lower() for lbl in labels]

    if match == "strict":
        gold_sets = [set(doc) for doc in gold_lists]
        pred_sets = [set(doc) for doc in pred_lists]
        return _compute_ner_counts_and_metrics(gold_sets, pred_sets, label_set)

    if match != "overlap":
        raise ValueError(f"Unsupported match mode: {match}")

    # Overlap/IoU matching (greedy per type)
    # We'll compute counts by building one-to-one matches within each type.
    per_type_counts: Dict[str, Dict[str, int]] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in label_set}
    tp = fp = fn = 0

    for gdoc, pdoc in zip(gold_lists, pred_lists):
        # Per-type partition
        by_type_gold: Dict[str, List[_Span]] = {
            t: [s for s in gdoc if s.typ == t] for t in label_set
        }
        by_type_pred: Dict[str, List[_Span]] = {
            t: [s for s in pdoc if s.typ == t] for t in label_set
        }

        for t in label_set:
            gold_t = by_type_gold[t]
            pred_t = by_type_pred[t]
            # Build all candidate pairs with IoU >= threshold
            pairs: List[Tuple[int, int, float]] = []
            for gi, gs in enumerate(gold_t):
                for pi, ps in enumerate(pred_t):
                    iou = gs.iou(ps)
                    if iou >= iou_threshold:
                        pairs.append((gi, pi, iou))
            # Greedy match by IoU descending
            pairs.sort(key=lambda x: x[2], reverse=True)
            matched_g = set()
            matched_p = set()
            for gi, pi, _ in pairs:
                if gi in matched_g or pi in matched_p:
                    continue
                matched_g.add(gi)
                matched_p.add(pi)
            tp_t = len(matched_g)
            fp_t = len(pred_t) - len(matched_p)
            fn_t = len(gold_t) - len(matched_g)

            per_type_counts[t]["tp"] += tp_t
            per_type_counts[t]["fp"] += fp_t
            per_type_counts[t]["fn"] += fn_t

            tp += tp_t
            fp += fp_t
            fn += fn_t

    micro_p = _safe_div(tp, tp + fp)
    micro_r = _safe_div(tp, tp + fn)
    micro_f1 = 0.0 if (micro_p + micro_r) == 0 else 2 * micro_p * micro_r / (micro_p + micro_r)

    per_type: Dict[str, Dict[str, float | int]] = {}
    for t, c in per_type_counts.items():
        p = _safe_div(c["tp"], c["tp"] + c["fp"])
        r = _safe_div(c["tp"], c["tp"] + c["fn"])
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        per_type[t] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
        }

    macro_p = _safe_div(sum(pt["precision"] for pt in per_type.values()), len(per_type) or 1)
    macro_r = _safe_div(sum(pt["recall"] for pt in per_type.values()), len(per_type) or 1)
    macro_f1 = _safe_div(sum(pt["f1"] for pt in per_type.values()), len(per_type) or 1)

    return {
        "micro": {
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_type": per_type,
    }
