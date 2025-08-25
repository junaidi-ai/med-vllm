#!/usr/bin/env python3
"""
Benchmark ontology linking performance and cache effectiveness on longer notes.

Usage:
  python benchmarks/benchmark_linking.py --paragraphs 50 --runs 3 --ontology RXNORM

It reports timings for the first and subsequent link_entities() runs and
lookup_in_ontology() cache stats across runs.
"""

from __future__ import annotations

import argparse
import time
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from medvllm.tasks import NERProcessor
from medvllm.tasks.ner_processor import lookup_in_ontology


def build_long_note(paragraphs: int) -> str:
    para = (
        "Patient with myocardial infarction presented to the ER. "
        "Aspirin 81 mg daily was given and metformin HCl was continued. "
        "Hemoglobin 13.2 g/dL and troponin elevated. Follow-up in 2 days.\n"
    )
    return para * max(1, paragraphs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark entity linking caching")
    parser.add_argument(
        "--paragraphs", type=int, default=30, help="Number of paragraphs to concatenate"
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of link_entities runs to time")
    parser.add_argument("--ontology", type=str, default="RXNORM", help="Ontology to link against")
    args = parser.parse_args()

    # Configure processor: enable extended gazetteer and entity linking
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        ner_enable_extended_gazetteer=True,
        entity_linking=SimpleNamespace(
            enabled=True, default_ontology=args.ontology, fuzzy_threshold=0.5
        ),
    )
    proc = NERProcessor(inference_pipeline=None, config=cfg)

    text = build_long_note(args.paragraphs)

    # Extract once (show spinner)
    with Progress(SpinnerColumn(), TextColumn("[bold]Extracting entities...")) as p:
        task = p.add_task("extract", total=None)
        t0 = time.perf_counter()
        res = proc.extract_entities(text)
        t1 = time.perf_counter()
        p.remove_task(task)

    # Clear cache before first linking timing
    lookup_in_ontology.cache_clear()

    timings = []
    cache_stats = []
    current = None
    runs = max(1, args.runs)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Linking entities[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("linking", total=runs)
        for _ in range(runs):
            s = time.perf_counter()
            current = proc.link_entities(res)  # uses default ontology
            e = time.perf_counter()
            timings.append(e - s)
            ci = lookup_in_ontology.cache_info()
            cache_stats.append((ci.hits, ci.misses, ci.currsize))
            progress.advance(task)
        progress.remove_task(task)

    print("=== Benchmark: Ontology Linking Cache Effectiveness ===")
    print(f"Paragraphs: {args.paragraphs}")
    print(f"Ontology:   {args.ontology}")
    print(f"Extract time: {t1 - t0:.4f}s; Entities: {len(current.entities) if current else 0}")
    for idx, (dt, cs) in enumerate(zip(timings, cache_stats), 1):
        h, m, sz = cs
        print(f"Run {idx}: {dt:.4f}s | cache hits={h}, misses={m}, size={sz}")


if __name__ == "__main__":
    main()
