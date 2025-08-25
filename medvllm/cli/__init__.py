"""Med-vLLM command line interface.

Provides a unified CLI with:
- model management commands under `model`
- inference commands under `inference` (NER, generate, classification)

Use `-h/--help` on any command for contextual guidance and examples.
"""

from typing import Any, Optional

import click
from click import Group
from rich.console import Console
from rich.traceback import install

# Enable rich traceback for better error messages
install(show_locals=True)

# Create console instance
console = Console()

# Global Click context settings for consistent help UX
CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 100,
}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
def cli() -> None:
    """Med-vLLM - Medical Variant of vLLM for Healthcare Applications.

    Tips:
    - Use `--text` for inline input, `--input` for files, or pipe via stdin.
    - Add `--json-out`/`--json-meta` to emit machine-readable outputs.

    Examples:
      python -m medvllm.cli model list
      python -m medvllm.cli inference ner --text "HTN on metformin" --json-out
      cat note.txt | python -m medvllm.cli inference generate --model your-hf-model
    """
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

if __name__ == "__main__":
    cli()  # type: ignore[no-untyped-call]
