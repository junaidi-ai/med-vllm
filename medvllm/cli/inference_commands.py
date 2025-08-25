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
import os

import click
from rich.console import Console
from rich.table import Table

from medvllm.tasks import NERProcessor, TextGenerator, MedicalConstraints
from medvllm.engine.model_runner.registry import get_registry, ModelNotFoundError

console = Console()


@click.group(name="inference")
def inference_group() -> None:
    """Run inference tasks (NER, text generation)."""
    pass


# -----------------------------
# Utilities
# -----------------------------


def _read_input(text: Optional[str], input_file: Optional[str], input_format: str = "auto") -> str:
    """Read input from direct text, file path (txt/pdf), or stdin.

    Args:
        text: Direct text input.
        input_file: Path to an input file.
        input_format: 'auto' | 'text' | 'pdf'. When 'auto', inferred from file ext.

    Returns:
        The input content as a string.
    """
    if text:
        return text
    if input_file:
        fmt = input_format.lower().strip() if input_format else "auto"
        ext = os.path.splitext(input_file)[1].lower()
        if fmt == "auto":
            if ext == ".pdf":
                fmt = "pdf"
            else:
                fmt = "text"

        if fmt == "pdf":
            try:
                # Lazy import to avoid hard dependency
                from pypdf import PdfReader  # type: ignore

                reader = PdfReader(input_file)
                pages = []
                for p in reader.pages:
                    try:
                        pages.append(p.extract_text() or "")
                    except Exception:
                        pages.append("")
                content = "\n\n".join(pages).strip()
                if not content:
                    raise click.ClickException(
                        "No extractable text found in PDF. Ensure the PDF is text-based, not scanned."
                    )
                return content
            except ImportError as e:
                raise click.ClickException(
                    "PDF support requires 'pypdf'. Install it (e.g., pip install pypdf) or use --input-format text."
                ) from e
        # default: treat as text file
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    # Fallback: read from stdin if piped
    if not click.get_text_stream("stdin").isatty():
        return click.get_text_stream("stdin").read()
    raise click.UsageError("No input provided. Use --text, --input <file>, or pipe via stdin.")


# -----------------------------
# Model/task validation
# -----------------------------


def _validate_model_task(model: Optional[str], task: str) -> None:
    """Validate that a model supports a given task based on registry metadata.

    If the model is not registered, we cannot validate and will warn (but allow).
    If registered and capabilities specify supported tasks, enforce compatibility.
    """
    if not model:
        return
    try:
        registry = get_registry()
        if not registry.is_registered(model):
            console.print(
                f"[yellow]Warning:[/] Model '{model}' is not registered; skipping compatibility validation."
            )
            return
        meta = registry.get_metadata(model)
        caps = getattr(meta, "capabilities", {}) or {}
        tasks = set(caps.get("tasks", []) or [])
        if tasks and task not in tasks:
            raise click.ClickException(
                f"Model '{model}' does not support task '{task}'. Supported tasks: {sorted(tasks)}"
            )
    except ModelNotFoundError:
        console.print(
            f"[yellow]Warning:[/] Model '{model}' not found in registry; skipping compatibility validation."
        )
    except click.ClickException:
        # Propagate validation errors to Click to signal command failure
        raise
    except Exception as e:
        # Do not block execution on validation errors; provide a clear message
        console.print(f"[yellow]Validation warning:[/] {e}")


# -----------------------------
# NER command
# -----------------------------


@inference_group.command(name="ner")
@click.option("--text", "text_", type=str, help="Direct input text.")
@click.option(
    "--input", "input_file", type=click.Path(exists=True, dir_okay=False), help="Input file path."
)
@click.option(
    "--input-format",
    type=click.Choice(["auto", "text", "pdf"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Input format when using --input. With auto, infers from file extension.",
)
@click.option(
    "--model", type=str, required=False, help="Optional model name or path for model-backed NER."
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
    input_format: str,
    model: Optional[str],
    ontology: str,
    json_out: bool,
    no_link: bool,
    output_file: Optional[str],
) -> None:
    """Run medical NER on input text and print entities (optionally linked)."""
    text = _read_input(text_, input_file, input_format)
    # Build processor and optional model-backed adapter
    if model:
        _validate_model_task(model, task="ner")
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
        except Exception as e:  # pragma: no cover
            raise click.ClickException(
                "Model-backed NER requires 'transformers'. Install it to use --model."
            ) from e

        try:
            nlp = hf_pipeline(
                "token-classification",
                model=model,
                aggregation_strategy="simple",
            )

            class HFTokenClsAdapter:
                def __init__(self, pipe: Any) -> None:
                    self.pipe = pipe

                def run_inference(self, text: str, task_type: str = "ner") -> dict:
                    if task_type != "ner":
                        return {"entities": []}
                    outs = self.pipe(text)
                    ents = []
                    for ent in outs:
                        start = int(ent.get("start", -1))
                        end = int(ent.get("end", -1))
                        word = ent.get("word")
                        if (start < 0 or end < 0) and isinstance(word, str):
                            # best-effort fallback
                            idx = text.find(word)
                            if idx >= 0:
                                start, end = idx, idx + len(word)
                        etype = ent.get("entity_group") or ent.get("entity") or "entity"
                        ents.append(
                            {
                                "text": word
                                if isinstance(word, str)
                                else (text[start:end] if start >= 0 and end >= 0 else ""),
                                "type": str(etype).lower(),
                                "start": start,
                                "end": end,
                                "confidence": float(ent.get("score", 0.0)),
                            }
                        )
                    return {"entities": ents}

            adapter = HFTokenClsAdapter(nlp)
            proc = NERProcessor(inference_pipeline=adapter, config=None)
        except Exception as e:
            raise click.ClickException(f"Model-backed NER failed: {e}") from e
    else:
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
@click.option(
    "--input-format",
    type=click.Choice(["auto", "text", "pdf"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Input format when using --input. With auto, infers from file extension.",
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
    input_format: str,
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
    # Validate model/task compatibility if possible
    _validate_model_task(model, task="generation")
    prompt = _read_input(text_, input_format=input_format, input_file=input_file)

    constraints = MedicalConstraints()
    if no_disclaimer:
        constraints.enforce_disclaimer = False
    # 0 values mean disabled
    if target_chars and target_chars > 0:
        constraints.max_length_chars = target_chars
    if target_words and target_words > 0:
        constraints.target_word_count = target_words

    generator = TextGenerator(model, constraints=constraints)
    # Minor validation/warnings for strategy/parameters
    if strategy.lower() == "greedy" and beam_width and beam_width != 1:
        console.print("[yellow]Warning:[/] beam_width is ignored for greedy strategy.")
    if strategy.lower() == "beam" and (top_k or (top_p and top_p > 0)):
        console.print("[yellow]Warning:[/] top-k/top-p are typically ignored for beam search.")

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
    "--input-format",
    type=click.Choice(["auto", "text", "pdf"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Input format when using --input. With auto, infers from file extension.",
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
    input_format: str,
    model: str,
    json_out: bool,
    output_file: Optional[str],
) -> None:
    """Run text classification using Hugging Face pipeline if available.

    This is a lightweight wrapper to enable basic classification via CLI until a
    dedicated classifier exists in medvllm/tasks.
    """
    text = _read_input(text_, input_file, input_format)
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
