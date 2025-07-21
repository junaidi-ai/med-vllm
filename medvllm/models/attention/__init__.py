"""Attention mechanisms for medical language models.

This package provides custom attention implementations optimized for medical NLP tasks,
including multi-head attention with various optimizations and specialized attention patterns.
"""

from .medical_attention import MedicalMultiheadAttention, flash_attention_forward

__all__ = [
    "MedicalMultiheadAttention",
    "flash_attention_forward",
]
