"""Medical Model Adapters for Nano vLLM.

This module provides adapters for medical language models including BioBERT and ClinicalBERT,
with support for tensor parallelism, CUDA optimization, and medical domain-specific features.

The adapters are now modularized in the adapters/ subdirectory for better maintainability.
"""

from typing import Any, Dict, Type, Union

import torch.nn as nn
from transformers import PreTrainedModel

# Import from modular adapter structure
from .adapters import BioBERTAdapter, ClinicalBERTAdapter
from .adapters.medical_adapter_base import MedicalModelAdapterBase


def create_medical_adapter(
    model_type: str, model: Union[nn.Module, PreTrainedModel], config: Dict[str, Any]
) -> "MedicalModelAdapterBase":
    """Factory function to create a medical model adapter.

    Args:
        model_type: Type of medical model (biobert, clinicalbert)
        model: Underlying model to adapt (can be a PyTorch Module or HuggingFace PreTrainedModel)
        config: Configuration dictionary for the adapter

    Returns:
        Created medical model adapter

    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()

    if model_type in ["biobert", "bio_bert", "dmis-lab/biobert"]:
        # Cast to nn.Module to satisfy mypy
        return BioBERTAdapter(model=model, config=config)
    elif model_type in [
        "clinicalbert",
        "clinical_bert",
        "emilyalsentzer/bio_clinicalbert",
    ]:
        # Cast to nn.Module to satisfy mypy
        return ClinicalBERTAdapter(model=model, config=config)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: biobert, clinicalbert"
        )


# Export all adapter classes for backward compatibility
__all__ = [
    "MedicalModelAdapter",
    "BioBERTAdapter",
    "ClinicalBERTAdapter",
    "create_medical_adapter",
]
