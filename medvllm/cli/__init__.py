"""Med-vLLM command line interface.

Provides a unified CLI with:
- model management commands under `model`
- inference commands under `inference` (NER, generate, classification)
- examples command for quick-start usage samples

Use `-h/--help` on any command for contextual guidance and examples.
"""

from typing import Any, Optional

import click
from click import Group
from rich.console import Console
from rich.traceback import install
from medvllm.cli.utils import set_verbose, console

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

    def register_commands(cli: Any) -> None:
        """Register all CLI commands.

        Args:
            cli: The Click CLI group to register commands with
        """
        register_model_commands(cli)
        register_inference_commands(cli)

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


if __name__ == "__main__":
    cli()  # type: ignore[no-untyped-call]
