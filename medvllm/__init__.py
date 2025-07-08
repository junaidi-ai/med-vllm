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
def main():
    """Entry point for the medvllm CLI."""
    from .cli import cli
    sys.exit(cli())


class DummyModule:
    """Dummy module that raises an informative error when accessed."""

    def __init__(self, name: str, pkg: str):
        self._name = name
        self._pkg = pkg

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            f"{self._name} requires {self._pkg} which is not installed. "
            f"Please install it with: pip install {self._pkg}"
        )

    def __getattr__(self, name: str) -> Any:
        raise ImportError(
            f"{self._name} requires {self._pkg} which is not installed. "
            f"Please install it with: pip install {self._pkg}"
        )


def lazy_import(name: str, pkg: str) -> Any:
    """Lazily import a module or return a dummy if not available."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return DummyModule(name, pkg)


# Make the package available for import
__all__ = ["LLM", "SamplingParams"]

# Lazy imports for PyTorch-dependent modules
if HAS_TORCH:
    try:
        from .llm import LLM
        from .sampling_params import SamplingParams
    except ImportError as e:
        if "torch" in str(e).lower():
            LLM = DummyModule("LLM", "torch")  # type: ignore
            SamplingParams = DummyModule("SamplingParams", "torch")  # type: ignore
        else:
            raise
else:
    LLM = DummyModule("LLM", "torch")  # type: ignore
    SamplingParams = DummyModule("SamplingParams", "torch")  # type: ignore


def __getattr__(name: str) -> Any:
    """Lazy import of modules to prevent circular imports."""
    if name == "LLM":
        return LLM
    elif name == "SamplingParams":
        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
