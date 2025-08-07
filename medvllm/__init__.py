"""
MedVLLM - Medical Variant of the vLLM library for medical NLP tasks.
"""

import importlib
import os
import sys
from typing import Any, TypeVar

# Version of the package
__version__ = "0.1.0"  # Default version, can be overridden by setuptools

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
__all__ = [
    "LLM",
    "SamplingParams",
    "MedicalModelAdapter",
    "BioBERTAdapter",
    "ClinicalBERTAdapter",
    "AdapterManager",
    "models",
    "tokenizers",
]

# Lazy imports for PyTorch-dependent modules
if HAS_TORCH:
    try:
        from .llm import LLM
        from .models.adapter import (
            BioBERTAdapter,
            ClinicalBERTAdapter,
        )
        from .models.adapter_manager import AdapterManager
        from .models.adapters.base import MedicalModelAdapterBase
        from .sampling_params import SamplingParams
        
        # Import models module
        from . import models
        
        # Create a dummy tokenizers module for compatibility
        class DummyTokenizerModule:
            """Dummy tokenizers module for compatibility."""
            pass
            
        tokenizers = DummyTokenizerModule()
        
    except ImportError as e:
        if "torch" in str(e).lower():
            LLM = DummyModule("LLM", e)  # type: ignore
            SamplingParams = DummyModule("SamplingParams", e)  # type: ignore
            MedicalModelAdapter = DummyModule("MedicalModelAdapter", e)  # type: ignore
            BioBERTAdapter = DummyModule("BioBERTAdapter", e)  # type: ignore
            ClinicalBERTAdapter = DummyModule("ClinicalBERTAdapter", e)  # type: ignore
            AdapterManager = DummyModule("AdapterManager", e)  # type: ignore
            models = DummyModule("models", e)  # type: ignore
            tokenizers = DummyModule("tokenizers", e)  # type: ignore
        else:
            raise
else:
    LLM = DummyModule("LLM", ImportError("torch is not installed"))  # type: ignore
    SamplingParams = DummyModule("SamplingParams", ImportError("torch is not installed"))  # type: ignore
    MedicalModelAdapter = DummyModule("MedicalModelAdapter", ImportError("torch is not installed"))  # type: ignore
    BioBERTAdapter = DummyModule("BioBERTAdapter", ImportError("torch is not installed"))  # type: ignore
    ClinicalBERTAdapter = DummyModule("ClinicalBERTAdapter", ImportError("torch is not installed"))  # type: ignore
    AdapterManager = DummyModule("AdapterManager", ImportError("torch is not installed"))  # type: ignore
    models = DummyModule("models", ImportError("torch is not installed"))  # type: ignore
    tokenizers = DummyModule("tokenizers", ImportError("torch is not installed"))  # type: ignore


def __getattr__(name: str) -> Any:
    """Lazy import of modules to prevent circular imports."""
    if name == "LLM":
        return LLM
    elif name == "SamplingParams":
        return SamplingParams
    elif name == "MedicalModelAdapter":
        return MedicalModelAdapter
    elif name == "BioBERTAdapter":
        return BioBERTAdapter
    elif name == "ClinicalBERTAdapter":
        return ClinicalBERTAdapter
    elif name == "AdapterManager":
        return AdapterManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
