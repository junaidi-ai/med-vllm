"""Med-vLLM command line interface.

Provides a unified CLI with:
- model management commands under `model`
- inference commands under `inference` (NER, generate, classification)
- examples command for quick-start usage samples

Use `-h/--help` on any command for contextual guidance and examples.
"""

from typing import Any, Optional
from pathlib import Path

import click
from click import Group
from rich.console import Console
from rich.traceback import install
from medvllm.cli.utils import set_verbose, console
from medvllm.configs.profiles import _default_profiles_dir, load_profile, profile_engine_kwargs
from rich.table import Table

# Enable rich traceback for better error messages
install(show_locals=True)

# Console is provided by utils for shared styling

# Global Click context settings for consistent help UX
CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 100,
}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose debug output for operations (also honors MEDVLLM_VERBOSE=1)",
    envvar="MEDVLLM_VERBOSE",
)
def cli(verbose: bool = False) -> None:
    """Med-vLLM - Medical Variant of vLLM for Healthcare Applications.

    Tips:
    - Use `--text` for inline input, `--input` for files, or pipe via stdin.
    - Add `--json-out`/`--json-meta` to emit machine-readable outputs.

    Examples:
      python -m medvllm.cli model list
      python -m medvllm.cli inference ner --text "HTN on metformin" --json-out
      cat note.txt | python -m medvllm.cli inference generate --model your-hf-model
    """
    # Set global verbosity once at entry
    set_verbose(bool(verbose))

    # no-op group body
    pass


# Import and register subcommands
try:
    from .model_commands import register_commands as register_model_commands
    from .inference_commands import register_commands as register_inference_commands
    from .training_commands import register_commands as register_training_commands
    from .validation_commands import validate_group

    def register_commands(cli: Any) -> None:
        """Register all CLI commands.

        Args:
            cli: The Click CLI group to register commands with
        """
        register_model_commands(cli)
        register_inference_commands(cli)
        register_training_commands(cli)
        # Add validation commands (accuracy/statistical equivalence)
        cli.add_command(validate_group)

except ImportError as e:
    console.print(f"[yellow]Warning: Could not load CLI commands: {e}[/]")

    # Fallback implementation if imports fail
    def register_commands(cli: Any) -> None:
        """Dummy implementation when CLI commands can't be loaded."""
        pass


register_commands(cli)


@cli.command()
def examples() -> None:
    """Show common example commands for medical NLP tasks."""
    console.print("[bold]Quick examples[/bold]\n")
    lines = [
        ("List registered models", "python -m medvllm.cli model list"),
        ("Show model capabilities", "python -m medvllm.cli model list-capabilities"),
        (
            "Register a model",
            "python -m medvllm.cli model register biobert /models/biobert --type generic",
        ),
        (
            "Run NER on text (JSON)",
            "python -m medvllm.cli inference ner --text 'HTN and DM' --json-out",
        ),
        (
            "Run NER on file and save",
            "python -m medvllm.cli inference ner --input note.txt --json-out --output ner.json",
        ),
        (
            "Generate text (beam)",
            "python -m medvllm.cli inference generate --text 'Explain hypertension' --model gpt2 --strategy beam --beam-width 3",
        ),
        (
            "Generate with metadata",
            "cat prompt.txt | python -m medvllm.cli inference generate --model gpt2 --json-meta --output out.txt",
        ),
        (
            "Classify sentiment",
            "python -m medvllm.cli inference classification --text 'This is helpful' --json-out",
        ),
        (
            "Verbose mode",
            "MEDVLLM_VERBOSE=1 python -m medvllm.cli inference ner --text 'on metformin' --ontology UMLS --verbose",
        ),
    ]
    for title, cmd in lines:
        console.print(f"• [cyan]{title}[/cyan]\n    [dim]{cmd}[/dim]")


@cli.command(name="list-profiles")
@click.option("--json", "json_out", is_flag=True, help="Emit JSON instead of a table.")
def list_profiles(json_out: bool = False) -> None:
    """List available deployment profiles from configs/deployment/.

    Shows profile name, category, and key engine defaults. Use --json for machine-readable output.
    """
    try:
        prof_dir = _default_profiles_dir()
    except Exception as e:
        console.print(f"[red]Failed to locate profiles directory:[/] {e}")
        raise click.ClickException(str(e))

    entries = []
    for p in sorted(Path(prof_dir).glob("*.json")):
        try:
            raw = load_profile(p)
            eng = profile_engine_kwargs(raw)
            name = str(raw.get("profile") or p.stem)
            cat = raw.get("category") or ""
            desc = raw.get("description") or ""
            entries.append(
                {
                    "name": name,
                    "category": cat,
                    "description": desc,
                    # Key engine defaults commonly used for quick comparison
                    "tensor_parallel_size": eng.get("tensor_parallel_size"),
                    "quantization_bits": eng.get("quantization_bits"),
                    "quantization_method": eng.get("quantization_method"),
                    "enable_mixed_precision": eng.get("enable_mixed_precision"),
                    "mixed_precision_dtype": eng.get("mixed_precision_dtype"),
                    "attention_impl": eng.get("attention_impl"),
                    "enable_flash_attention": eng.get("enable_flash_attention"),
                    "enable_memory_pooling": eng.get("enable_memory_pooling"),
                    "pool_device": eng.get("pool_device"),
                }
            )
        except Exception as e:
            # Continue listing; report problematic file
            entries.append(
                {
                    "name": p.stem,
                    "category": "",
                    "description": f"Error: {e}",
                }
            )

    if json_out:
        import json as _json

        console.print_json(data={"profiles": entries})
        return

    table = Table(title="Deployment Profiles", expand=True)
    table.add_column("Name", no_wrap=True, overflow="fold")
    table.add_column("Category", no_wrap=True)
    table.add_column("TP")
    table.add_column("Quant")
    table.add_column("MP")
    table.add_column("Attention")
    table.add_column("MemPool")

    for e in entries:
        q = ""
        if e.get("quantization_bits"):
            q = f"{e.get('quantization_bits')} ({e.get('quantization_method') or ''})".strip()
        mp = "on" if e.get("enable_mixed_precision") else "off"
        if e.get("mixed_precision_dtype"):
            mp = f"{mp}:{e.get('mixed_precision_dtype')}"
        att_impl = e.get("attention_impl") or "auto"
        fa = e.get("enable_flash_attention")
        if fa is True and att_impl != "flash":
            att = f"{att_impl}+flash"
        else:
            att = att_impl
        mem = "on" if e.get("enable_memory_pooling") else "off"
        if e.get("pool_device"):
            mem = f"{mem}:{e.get('pool_device')}"

        table.add_row(
            str(e.get("name", "")),
            str(e.get("category", "")),
            str(e.get("tensor_parallel_size") or 1),
            q,
            mp,
            att,
            mem,
        )

    console.print(table)


if __name__ == "__main__":
    cli()  # type: ignore[no-untyped-call]
