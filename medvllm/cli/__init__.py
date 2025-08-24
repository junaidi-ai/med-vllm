"""Med-vLLM command line interface."""

from typing import Any, Optional

import click
from click import Group
from rich.console import Console
from rich.traceback import install

# Enable rich traceback for better error messages
install(show_locals=True)

# Create console instance
console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """Med-vLLM - Medical Variant of vLLM for Healthcare Applications."""
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
