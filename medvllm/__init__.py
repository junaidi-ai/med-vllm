"""
MedVLLM - Medical Variant of the vLLM library for medical NLP tasks.
"""

import importlib
import sys
from typing import Any, TypeVar

# Check for required dependencies
HAS_TORCH = False
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    pass

# Define a type variable for generic type hints
T = TypeVar("T")


# CLI entry point
def main() -> None:
    """Entry point for the medvllm CLI."""
    from .cli import cli

    sys.exit(cli())


class DummyModule:
    """Dummy module that raises an ImportError when used."""

    def __init__(self, name: str, error: ImportError) -> None:
        self.name = name
        self.error = error

    def __getattr__(self, name: str) -> None:
        raise ImportError(
            f"{self.name} is required but could not be imported: {self.error}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            f"{self.name} is required but could not be imported: {self.error}"
        )


def lazy_import(name: str, pkg: str) -> Any:
    """Lazily import a module or return a dummy if not available."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        return DummyModule(name, e)


# Make the package available for import
__all__ = ["LLM", "SamplingParams"]

# Lazy imports for PyTorch-dependent modules
if HAS_TORCH:
    try:
        from .llm import LLM
        from .sampling_params import SamplingParams
    except ImportError as e:
        if "torch" in str(e).lower():
            LLM = DummyModule("LLM", e)  # type: ignore
            SamplingParams = DummyModule("SamplingParams", e)  # type: ignore
        else:
            raise
else:
    LLM = DummyModule("LLM", ImportError("torch is not installed"))  # type: ignore
    SamplingParams = DummyModule("SamplingParams", ImportError("torch is not installed"))  # type: ignore


def __getattr__(name: str) -> Any:
    """Lazy import of modules to prevent circular imports."""
    if name == "LLM":
        return LLM
    elif name == "SamplingParams":
        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
