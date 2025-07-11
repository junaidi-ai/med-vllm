"""Optimization techniques for medical models."""

from .flash_attention import FlashAttentionConfig, enable_flash_attention
from .quantization import QuantizationConfig, quantize_model

__all__ = [
    "quantize_model",
    "QuantizationConfig",
    "enable_flash_attention",
    "FlashAttentionConfig",
]
