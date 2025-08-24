"""Inference-related CLI commands for Med-vLLM.

Provides an `inference` command group with subcommands:
- ner: run lightweight medical NER via `NERProcessor`
- generate: run text generation via `TextGenerator`

This mirrors the style of `model_commands.py` (Click-based) and plugs into the
main CLI via a `register_commands(cli)` function.
"""

from __future__ import annotations

from typing import Any, Optional
import json

import click
from rich.console import Console
from rich.table import Table

from medvllm.tasks import NERProcessor, TextGenerator, MedicalConstraints

console = Console()


@click.group(name="inference")
def inference_group() -> None:
    """Run inference tasks (NER, text generation)."""
    pass


# -----------------------------
# Utilities
# -----------------------------


def _read_input(text: Optional[str], input_file: Optional[str]) -> str:
    if text:
        return text
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback: read from stdin if piped
    if not click.get_text_stream("stdin").isatty():
        return click.get_text_stream("stdin").read()
    raise click.UsageError("No input provided. Use --text, --input <file>, or pipe via stdin.")


# -----------------------------
# NER command
# -----------------------------


@inference_group.command(name="ner")
@click.option("--text", "text_", type=str, help="Direct input text.")
@click.option(
    "--input", "input_file", type=click.Path(exists=True, dir_okay=False), help="Input file path."
)
@click.option(
    "--ontology", type=str, default="UMLS", show_default=True, help="Ontology to use for linking."
)
@click.option("--json-out", is_flag=True, help="Output JSON instead of a table.")
@click.option("--no-link", is_flag=True, help="Disable ontology linking.")
@click.option(
    "--output", "output_file", type=click.Path(dir_okay=False), help="Write output to file."
)
def cmd_ner(
    text_: Optional[str],
    input_file: Optional[str],
    ontology: str,
    json_out: bool,
    no_link: bool,
    output_file: Optional[str],
) -> None:
    """Run medical NER on input text and print entities (optionally linked)."""
    text = _read_input(text_, input_file)

    proc = NERProcessor(inference_pipeline=None, config=None)
    ner = proc.extract_entities(text)

    if not no_link:
        try:
            ner = proc.link_entities(ner, ontology=ontology)
        except Exception as e:  # graceful degradation
            console.print(f"[yellow]Linking failed: {e}. Showing unlinked entities.[/]")

    if json_out:
        data = {"entities": ner.entities}
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        console.print_json(data=data)
        return

    table = Table(title="NER Entities")
    table.add_column("Text")
    table.add_column("Type")
    table.add_column("Span")
    table.add_column("Link (top)")

    for ent in ner.entities:
        span = f"{ent.get('start','?')}–{ent.get('end','?')}"
        link_disp = ""
        links = ent.get("ontology_links") or []
        if links:
            l0 = links[0]
            code = l0.get("code") or ""
            ont = l0.get("ontology") or ""
            name = l0.get("name") or ""
            link_disp = f"{ont}:{code} {name}".strip()
        table.add_row(str(ent.get("text", "")), str(ent.get("type", "")), span, link_disp)

    # Print table and optionally write to file
    console.print(table)
    if output_file:
        from io import StringIO

        buf = StringIO()
        tmp_console = Console(
            file=buf, force_terminal=False, color_system=None, width=console.width
        )
        tmp_console.print(table)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())


# -----------------------------
# Text generation command
# -----------------------------


@inference_group.command(name="generate")
@click.option("--text", "text_", type=str, help="Prompt text.")
@click.option(
    "--input", "input_file", type=click.Path(exists=True, dir_okay=False), help="Prompt file path."
)
@click.option("--model", type=str, required=True, help="Model name or path for generation.")
@click.option(
    "--strategy",
    type=click.Choice(["greedy", "sampling", "beam"], case_sensitive=False),
    default="beam",
    show_default=True,
)
@click.option(
    "--max-length", type=int, default=200, show_default=True, help="Max generation length (tokens)."
)
@click.option("--temperature", type=float, default=0.7, show_default=True)
@click.option("--top-p", type=float, default=0.9, show_default=True)
@click.option("--top-k", type=int, default=0, show_default=True, help="Top-k cutoff; 0 disables.")
@click.option("--beam-width", type=int, default=3, show_default=True)
@click.option(
    "--purpose", type=str, default=None, help="Profile purpose: patient | clinical | research."
)
@click.option("--readability", type=str, default=None, help="Readability: general | specialist.")
@click.option("--tone", type=str, default=None, help="Tone: formal | informal.")
@click.option("--structure", type=str, default=None, help="Structure: soap | bullet | paragraph.")
@click.option("--specialty", type=str, default=None, help="Specialty domain e.g., cardiology.")
@click.option(
    "--target-words",
    type=int,
    default=0,
    show_default=True,
    help="Approximate word target; 0 disables.",
)
@click.option(
    "--target-chars", type=int, default=0, show_default=True, help="Hard char limit; 0 disables."
)
@click.option("--no-disclaimer", is_flag=True, help="Disable enforced disclaimer in output.")
@click.option("--json-meta", is_flag=True, help="Print metadata JSON after text.")
@click.option(
    "--output",
    "output_file",
    type=click.Path(dir_okay=False),
    help="Write generated text to file; when --json-meta is set, writes metadata to <output>.meta.json as well.",
)
def cmd_generate(
    text_: Optional[str],
    input_file: Optional[str],
    model: str,
    strategy: str,
    max_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    beam_width: int,
    purpose: Optional[str],
    readability: Optional[str],
    tone: Optional[str],
    structure: Optional[str],
    specialty: Optional[str],
    target_words: int,
    target_chars: int,
    no_disclaimer: bool,
    json_meta: bool,
    output_file: Optional[str],
) -> None:
    """Generate medical text according to the chosen strategy and constraints."""
    prompt = _read_input(text_, input_file)

    constraints = MedicalConstraints()
    if no_disclaimer:
        constraints.enforce_disclaimer = False
    # 0 values mean disabled
    if target_chars and target_chars > 0:
        constraints.max_length_chars = target_chars
    if target_words and target_words > 0:
        constraints.target_word_count = target_words

    generator = TextGenerator(model, constraints=constraints)
    res = generator.generate(
        prompt,
        max_length=max_length,
        strategy=strategy,
        temperature=temperature,
        top_p=top_p if top_p > 0 else None,
        top_k=top_k if top_k > 0 else None,
        beam_width=beam_width,
        purpose=purpose,
        readability=readability,
        tone=tone,
        structure=structure,
        specialty=specialty,
        target_words=target_words if target_words > 0 else None,
        target_chars=target_chars if target_chars > 0 else None,
    )

    # Print generated text and optionally write to file
    console.print(res.generated_text)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(res.generated_text)
    if json_meta:
        meta_json = json.dumps(res.metadata, indent=2)
        if output_file:
            with open(f"{output_file}.meta.json", "w", encoding="utf-8") as f:
                f.write(meta_json)
        # Use plain print to avoid rich formatting in JSON
        print(meta_json)


def register_commands(cli: Any) -> None:
    """Register the inference command group with the main CLI group."""
    cli.add_command(inference_group)


# -----------------------------
# Classification command (optional deps)
# -----------------------------


@inference_group.command(name="classification")
@click.option("--text", "text_", type=str, help="Input text to classify.")
@click.option(
    "--input", "input_file", type=click.Path(exists=True, dir_okay=False), help="Input file path."
)
@click.option(
    "--model",
    type=str,
    default="distilbert-base-uncased-finetuned-sst-2-english",
    show_default=True,
    help="HF model id for text classification.",
)
@click.option("--json-out", is_flag=True, help="Output JSON instead of a table.")
@click.option(
    "--output", "output_file", type=click.Path(dir_okay=False), help="Write JSON result to file."
)
def cmd_classification(
    text_: Optional[str],
    input_file: Optional[str],
    model: str,
    json_out: bool,
    output_file: Optional[str],
) -> None:
    """Run text classification using Hugging Face pipeline if available.

    This is a lightweight wrapper to enable basic classification via CLI until a
    dedicated classifier exists in medvllm/tasks.
    """
    text = _read_input(text_, input_file)
    try:
        from transformers import pipeline  # type: ignore
    except Exception as e:
        raise click.ClickException(
            "transformers is not available for classification. Install extras or specify a simpler mode."
        ) from e

    try:
        clf = pipeline("text-classification", model=model, return_all_scores=False)
        result = clf(text)
    except Exception as e:  # pragma: no cover - depends on env
        raise click.ClickException(f"Failed to run classification: {e}") from e

    # Normalize result
    if isinstance(result, list) and result and isinstance(result[0], dict):
        out = {"label": result[0].get("label"), "score": float(result[0].get("score", 0.0))}
    elif isinstance(result, dict):
        out = {"label": result.get("label"), "score": float(result.get("score", 0.0))}
    else:
        out = {"label": None, "score": None, "raw": result}

    if json_out:
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        console.print_json(data=out)
        return

    tbl = Table(title="Classification Result")
    tbl.add_column("Label")
    tbl.add_column("Score")
    lbl = str(out.get("label"))
    scr = f"{out.get('score'):.4f}" if isinstance(out.get("score"), float) else ""
    tbl.add_row(lbl, scr)
    console.print(tbl)
    if output_file:
        # For file output, always write JSON for easier downstream use
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
