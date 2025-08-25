"""CLI utilities for consistent message styling, verbosity, and progress."""

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from rich.console import Console

# Shared console instance for all CLI modules
console = Console()

# Global verbosity flag controlled by root CLI
_VERBOSE: bool = False


def set_verbose(enabled: bool) -> None:
    """Enable or disable verbose logging for CLI commands."""
    global _VERBOSE
    _VERBOSE = bool(enabled)


def is_verbose() -> bool:
    """Return current verbose state."""
    return _VERBOSE


def debug(message: str) -> None:
    """Print a verbose/debug message when verbosity is enabled."""
    if _VERBOSE:
        # Escape square brackets for Rich markup using double brackets
        console.print(f"[dim][[debug]][/dim] {message}")


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


@contextmanager
def spinner(message: str) -> Iterator[None]:
    """A lightweight spinner for long-running operations.

    Usage:
        with spinner("Loading model..."):
            load_model()
    """
    with console.status(message + " ", spinner="dots"):
        yield


@contextmanager
def timed(section: str) -> Iterator[None]:
    """Context manager that logs elapsed time when verbose is enabled."""
    start = perf_counter()
    try:
        yield
    finally:
        dur = perf_counter() - start
        debug(f"{section} took {dur:.2f}s")
