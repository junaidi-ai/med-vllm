"""CLI utilities for consistent message styling and console access."""

from rich.console import Console

# Shared console instance for all CLI modules
console = Console()


def warn(message: str) -> None:
    """Print a standardized warning message."""
    console.print(f"[yellow]Warning:[/] {message}")


def error(message: str) -> None:
    """Print a standardized error message prefix (for non-Click exceptions)."""
    console.print(f"[red]✗[/] {message}")


def success(message: str) -> None:
    """Print a standardized success message."""
    console.print(f"[green]✓[/] {message}")


def hint(message: str) -> None:
    """Print a hint/suggestion line."""
    console.print(f"Hint: {message}")
