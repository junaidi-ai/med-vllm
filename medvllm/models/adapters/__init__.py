"""Medical model adapters for Nano vLLM.

This package provides modular medical language model adapters including:
- Base adapter interface (MedicalModelAdapter)
- BioBERT adapter for biomedical NLP
- ClinicalBERT adapter for clinical NLP
"""

from .base import MedicalModelAdapter as MedicalModelAdapter
from .biobert import BioBERTAdapter
from .clinicalbert import ClinicalBERTAdapter

__all__ = ["MedicalModelAdapter", "BioBERTAdapter", "ClinicalBERTAdapter"]
