#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List


CRITERIA = ["accuracy", "clarity", "completeness", "safety", "tone_style"]


def aggregate(path: str) -> Dict[str, Dict[str, float]]:
    per_strategy: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strat = row.get("strategy", "").strip()
            if not strat:
                continue
            scores = {}
            for c in CRITERIA:
                try:
                    scores[c] = float(row.get(c, "0") or 0)
                except ValueError:
                    scores[c] = 0.0
            per_strategy[strat].append(scores)

    summary: Dict[str, Dict[str, float]] = {}
    for strat, rows in per_strategy.items():
        agg = {c: 0.0 for c in CRITERIA}
        if not rows:
            summary[strat] = agg
            continue
        for r in rows:
            for c in CRITERIA:
                agg[c] += r.get(c, 0.0)
        for c in CRITERIA:
            agg[c] /= len(rows)
        agg["overall"] = sum(agg[c] for c in CRITERIA) / len(CRITERIA)
        summary[strat] = agg
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate expert evaluation CSV scores")
    ap.add_argument("path", help="Path to expert_eval CSV (see docs/expert_eval_template.csv)")
    args = ap.parse_args()
    summary = aggregate(args.path)
    for strat, scores in summary.items():
        print(strat)
        for k, v in scores.items():
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
