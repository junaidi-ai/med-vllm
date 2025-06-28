"""
Validation utilities for medical model configurations.

This module contains validation functions for medical model configurations.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .medical_config import MedicalModelConfig


class MedicalConfigValidator:
    """Validator for medical model configurations."""

    @staticmethod
    def validate_tensor_parallel_size(value: int) -> None:
        """Validate tensor_parallel_size parameter."""
        if not (1 <= value <= 8):
            raise ValueError(
                f"tensor_parallel_size must be between 1 and 8, got {value}"
            )

    @staticmethod
    def validate_entity_linking(config: "MedicalModelConfig") -> None:
        """Validate entity linking configuration."""
        if config.entity_linking_enabled and not config.entity_linking_knowledge_bases:
            raise ValueError(
                "Entity linking is enabled but no knowledge bases are specified"
            )

    @staticmethod
    def validate_medical_parameters(config: "MedicalModelConfig") -> None:
        """Validate all medical-specific parameters."""
        if (
            hasattr(config, "tensor_parallel_size")
            and config.tensor_parallel_size is not None
        ):
            MedicalConfigValidator.validate_tensor_parallel_size(
                config.tensor_parallel_size
            )

        if hasattr(config, "entity_linking_enabled") and config.entity_linking_enabled:
            MedicalConfigValidator.validate_entity_linking(config)

    @staticmethod
    def warn_deprecated(param_name: str, version: str, alternative: str = None) -> None:
        """Log a deprecation warning for a parameter."""
        msg = f"'{param_name}' is deprecated and will be removed in version {version}."
        if alternative:
            msg += f" Use '{alternative}' instead."
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
