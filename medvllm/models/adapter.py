"""Medical Model Adapters for Nano vLLM.

This module provides adapters for medical language models including BioBERT and ClinicalBERT,
with support for tensor parallelism, CUDA optimization, and medical domain-specific features.

The adapters are now modularized in the adapters/ subdirectory for better maintainability.
"""

from typing import Any, Dict, Union

import torch.nn as nn
from transformers import PreTrainedModel

# Import from modular adapter structure
from .adapters import BioBERTAdapter, ClinicalBERTAdapter, MedicalModelAdapterBase


def create_medical_adapter(
    model_type: Union[str, nn.Module, PreTrainedModel],
    model: Union[str, nn.Module, PreTrainedModel],
    config: Dict[str, Any],
) -> MedicalModelAdapterBase:
    """Factory function to create a medical model adapter.

    Args:
        model_type: Type of medical model (biobert, clinicalbert) OR the model if called with (model, model_type, config)
        model: Underlying model to adapt OR the model type string if called with (model, model_type, config)
        config: Configuration dictionary for the adapter

    Returns:
        Created medical model adapter

    Raises:
        ValueError: If model_type is not supported
    """
    # Support both call signatures:
    #   (model_type: str, model: nn.Module, config)
    #   (model: nn.Module, model_type: str, config)
    if isinstance(model_type, str) and not isinstance(model, str):
        adapter_type = model_type
        adapter_model = model  # type: ignore[assignment]
    else:
        # Called as (model, model_type, config)
        adapter_model = model_type  # type: ignore[assignment]
        adapter_type = model  # type: ignore[assignment]

    adapter_type = str(adapter_type).lower()

    if adapter_type in ["biobert", "bio_bert", "dmis-lab/biobert"]:
        # Cast to nn.Module to satisfy mypy
        return BioBERTAdapter(model=adapter_model, config=config)
    elif adapter_type in [
        "clinicalbert",
        "clinical_bert",
        "emilyalsentzer/bio_clinicalbert",
    ]:
        # Cast to nn.Module to satisfy mypy
        return ClinicalBERTAdapter(model=adapter_model, config=config)
    else:
        raise ValueError(
            f"Unsupported model type: {adapter_type}. " f"Supported types: biobert, clinicalbert"
        )


# Export all adapter classes for backward compatibility
__all__ = [
    "MedicalModelAdapterBase",
    "BioBERTAdapter",
    "ClinicalBERTAdapter",
    "create_medical_adapter",
]
