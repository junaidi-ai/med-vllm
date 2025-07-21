"""Medical model loaders and adapters for healthcare NLP."""

from .adapter import (
    BioBERTAdapter,
    ClinicalBERTAdapter,
    create_medical_adapter,
)
from .adapters.medical_adapter_base import MedicalModelAdapterBase
from .medical_models import BioBERTLoader, ClinicalBERTLoader

__all__ = [
    "BioBERTLoader",
    "ClinicalBERTLoader",
    "MedicalModelAdapter",
    "BioBERTAdapter",
    "ClinicalBERTAdapter",
    "create_medical_adapter",
]
