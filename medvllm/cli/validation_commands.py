from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import click

from medvllm.cli.utils import console
from medvllm.eval.validation import (
    ClassificationMetrics,
    compute_classification_metrics,
    mcnemar_test_equivalence,
    threshold_check,
)
from medvllm.eval.visualization import (
    plot_confusion_matrix_png,
    plot_pr_curve_png,
    plot_roc_curve_png,
)
from medvllm.eval.thresholds import DEFAULT_THRESHOLDS, load_thresholds_from_file


def _read_csv(
    path: str,
    *,
    y_true_col: str,
    y_pred_a_col: str,
    y_pred_b_col: str,
) -> Dict[str, List[str]]:
    import csv

    y_true: List[str] = []
    y_pred_a: List[str] = []
    y_pred_b: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(str(row[y_true_col]))
            y_pred_a.append(str(row[y_pred_a_col]))
            y_pred_b.append(str(row[y_pred_b_col]))
    return {"y_true": y_true, "y_pred_a": y_pred_a, "y_pred_b": y_pred_b}


@click.group(name="validate")
def validate_group() -> None:
    """Validation tools for accuracy and statistical equivalence."""


@validate_group.command(name="classification")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--y-true-col", required=True, help="CSV column for true labels")
@click.option("--y-pred-a-col", required=True, help="CSV column for baseline/original predictions")
@click.option("--y-pred-b-col", required=True, help="CSV column for optimized predictions")
@click.option("--out-dir", default="reports", show_default=True, type=click.Path(file_okay=False))
@click.option(
    "--threshold",
    multiple=True,
    type=str,
    help="Threshold in key=value form, e.g., accuracy=0.95 f1_macro=0.92",
)
@click.option(
    "--thresholds-file",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Load thresholds from JSON/YAML file. Can be a flat mapping or namespaced.",
)
@click.option(
    "--thresholds-preset",
    type=click.Choice(sorted(list(DEFAULT_THRESHOLDS.keys()))),
    required=False,
    help="Use a built-in thresholds preset (e.g., classification.general).",
)
@click.option(
    "--roc-scores-json",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Optional JSON file with probabilities for ROC/PR curves. Must have 'y_true' and 'scores_a'/'scores_b'.",
)
@click.option(
    "--prefix",
    default="classification_validation",
    show_default=True,
    help="Output filename prefix",
)
@click.option(
    "--signoff",
    multiple=True,
    type=str,
    help="Clinical sign-off metadata in key=value form (repeatable), e.g., reviewer=Dr.X approved=true notes='Looks good'",
)
@click.option(
    "--signoff-file",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Load sign-off metadata from JSON/YAML file.",
)
@click.option(
    "--fail-on-thresholds",
    is_flag=True,
    default=False,
    help="Exit with non-zero status if optimized model fails threshold checks.",
)
def classification_validate(
    csv_path: str,
    y_true_col: str,
    y_pred_a_col: str,
    y_pred_b_col: str,
    out_dir: str,
    threshold: List[str],
    thresholds_file: Optional[str],
    thresholds_preset: Optional[str],
    roc_scores_json: Optional[str],
    prefix: str,
    signoff: List[str],
    signoff_file: Optional[str],
    fail_on_thresholds: bool,
) -> None:
    """Validate optimized vs baseline classification outputs using McNemar's test and metrics.

    CSV must include columns for true labels and two sets of predictions.
    Writes JSON report and PNG plots to out_dir.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data = _read_csv(
        csv_path, y_true_col=y_true_col, y_pred_a_col=y_pred_a_col, y_pred_b_col=y_pred_b_col
    )
    y_true = data["y_true"]
    y_pred_a = data["y_pred_a"]
    y_pred_b = data["y_pred_b"]

    # Compute metrics
    metrics_a = compute_classification_metrics(y_true, y_pred_a)
    metrics_b = compute_classification_metrics(y_true, y_pred_b)

    # McNemar test
    mc = mcnemar_test_equivalence(y_true, y_pred_a, y_pred_b)

    # Thresholds (preset/file/inline). Merge with override precedence: inline > file/preset
    th_map: Dict[str, float] = {}
    # 1) Preset
    if thresholds_preset:
        th_map.update(DEFAULT_THRESHOLDS.get(thresholds_preset, {}))
    # 2) File
    if thresholds_file:
        try:
            loaded = load_thresholds_from_file(thresholds_file)
            # Allow either flat or namespaced structures
            if isinstance(loaded, dict):
                # If preset specified, and corresponding key exists, prefer that sub-dict
                if (
                    thresholds_preset
                    and thresholds_preset in loaded
                    and isinstance(loaded[thresholds_preset], dict)
                ):
                    th_map.update(
                        {
                            k: float(v)
                            for k, v in loaded[thresholds_preset].items()
                            if isinstance(v, (int, float))
                        }
                    )
                else:
                    th_map.update(
                        {k: float(v) for k, v in loaded.items() if isinstance(v, (int, float))}
                    )
        except Exception as e:
            console.print(f"[yellow]Warning: could not load thresholds file: {e}[/]")
    for item in threshold:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        try:
            th_map[k.strip()] = float(v.strip())
        except Exception:
            pass
    threshold_results_a = threshold_check(metrics_a, th_map) if th_map else {}
    threshold_results_b = threshold_check(metrics_b, th_map) if th_map else {}

    # Plots
    cm_path_a = str(Path(out_dir) / f"{prefix}_cm_baseline.png")
    cm_path_b = str(Path(out_dir) / f"{prefix}_cm_optimized.png")
    plot_confusion_matrix_png(
        metrics_a.confusion_matrix, metrics_a.labels, title="Baseline Confusion", out_path=cm_path_a
    )
    plot_confusion_matrix_png(
        metrics_b.confusion_matrix,
        metrics_b.labels,
        title="Optimized Confusion",
        out_path=cm_path_b,
    )

    roc_paths: Dict[str, Optional[str]] = {"a": None, "b": None}
    pr_paths: Dict[str, Optional[str]] = {"a": None, "b": None}
    if roc_scores_json:
        try:
            with open(roc_scores_json, "r", encoding="utf-8") as f:
                scores_payload = json.load(f)
            y_true_scores = scores_payload.get("y_true")
            scores_a = scores_payload.get("scores_a")
            scores_b = scores_payload.get("scores_b")
            if y_true_scores and scores_a:
                roc_paths["a"] = str(Path(out_dir) / f"{prefix}_roc_baseline.png")
                pr_paths["a"] = str(Path(out_dir) / f"{prefix}_pr_baseline.png")
                plot_roc_curve_png(y_true_scores, scores_a, out_path=roc_paths["a"])
                plot_pr_curve_png(y_true_scores, scores_a, out_path=pr_paths["a"])
            if y_true_scores and scores_b:
                roc_paths["b"] = str(Path(out_dir) / f"{prefix}_roc_optimized.png")
                pr_paths["b"] = str(Path(out_dir) / f"{prefix}_pr_optimized.png")
                plot_roc_curve_png(y_true_scores, scores_b, out_path=roc_paths["b"])
                plot_pr_curve_png(y_true_scores, scores_b, out_path=pr_paths["b"])
        except Exception as e:
            console.print(f"[yellow]Warning: could not render ROC/PR plots: {e}[/]")

    # Assemble report
    def _metrics_to_dict(m: ClassificationMetrics) -> Dict[str, Any]:
        return {
            "accuracy": m.accuracy,
            "precision_macro": m.precision_macro,
            "recall_macro": m.recall_macro,
            "f1_macro": m.f1_macro,
            "precision_micro": m.precision_micro,
            "recall_micro": m.recall_micro,
            "f1_micro": m.f1_micro,
            "labels": m.labels,
            "confusion_matrix": m.confusion_matrix,
        }

    # Sign-off metadata
    signoff_map: Dict[str, Any] = {}
    if signoff_file:
        try:
            with open(signoff_file, "r", encoding="utf-8") as f:
                text = f.read()
            try:
                signoff_map.update(json.loads(text))
            except Exception:
                try:
                    import yaml  # type: ignore

                    data = yaml.safe_load(text)
                    if isinstance(data, dict):
                        signoff_map.update(data)
                except Exception:
                    pass
        except Exception:
            pass
    for item in signoff:
        if "=" in item:
            k, v = item.split("=", 1)
            vv: Any = v.strip()
            if vv.lower() in {"true", "false"}:
                vv = vv.lower() == "true"
            signoff_map[k.strip()] = vv
    signoff_map.setdefault("date", datetime.utcnow().isoformat() + "Z")

    report: Dict[str, Any] = {
        "input_csv": csv_path,
        "metrics": {
            "baseline": _metrics_to_dict(metrics_a),
            "optimized": _metrics_to_dict(metrics_b),
        },
        "mcnemar": {
            "b": mc.b,
            "c": mc.c,
            "statistic": mc.statistic,
            "pvalue": mc.pvalue,
            "exact": mc.exact,
        },
        "thresholds": th_map,
        "threshold_results": {
            "baseline": threshold_results_a,
            "optimized": threshold_results_b,
        },
        "artifacts": {
            "cm_baseline_png": cm_path_a,
            "cm_optimized_png": cm_path_b,
            "roc_baseline_png": roc_paths["a"],
            "roc_optimized_png": roc_paths["b"],
            "pr_baseline_png": pr_paths["a"],
            "pr_optimized_png": pr_paths["b"],
        },
        "signoff": signoff_map,
    }

    out_json = str(Path(out_dir) / f"{prefix}_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    console.print("[green]Validation completed.[/]")
    console.print(f"Report: [cyan]{out_json}[/]")

    # Optional CI gate
    if fail_on_thresholds and th_map:
        # Gate on optimized results if thresholds provided
        if threshold_results_b and not all(bool(v) for v in threshold_results_b.values()):
            raise SystemExit(1)
