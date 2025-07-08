"""Med-vLLM command line interface."""

import click
from rich.console import Console
from rich.traceback import install

# Enable rich traceback for better error messages
install(show_locals=True)

# Create console instance
console = Console()

@click.group()
@click.version_option()
def cli():
    """Med-vLLM - Medical Variant of vLLM for Healthcare Applications."""
    pass

# Import and register subcommands
try:
    from .model_commands import register_commands
    register_commands(cli)
except ImportError as e:
    console.print(f"[yellow]Warning: Could not load CLI commands: {e}[/]")

if __name__ == "__main__":
    cli()
