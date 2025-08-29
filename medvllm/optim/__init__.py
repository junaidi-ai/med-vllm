"""Optimization techniques for medical models."""

from .flash_attention import FlashAttentionConfig, enable_flash_attention
from .quantization import (
    QuantizationConfig,
    quantize_model,
    bnb_load_quantized,
    bnb_save_stub,
    bnb_offline_hint,
)
from .medical_optimizer import MedicalModelOptimizer, OptimizerConfig

__all__ = [
    "quantize_model",
    "QuantizationConfig",
    "bnb_load_quantized",
    "bnb_save_stub",
    "bnb_offline_hint",
    "MedicalModelOptimizer",
    "OptimizerConfig",
    "enable_flash_attention",
    "FlashAttentionConfig",
]
