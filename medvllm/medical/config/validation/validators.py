"""
Validators for medical model configurations.

This module contains various validators for configuration values.
"""

import warnings
from typing import Any, Dict, Optional, TYPE_CHECKING

from .exceptions import ValidationError, FieldValueError

if TYPE_CHECKING:
    from ..base import BaseMedicalConfig


class MedicalConfigValidator:
    """Validator for medical model configurations."""

    @staticmethod
    def validate_tensor_parallel_size(value: int) -> None:
        """Validate tensor_parallel_size parameter.
        
        Args:
            value: The value to validate
            
        Raises:
            FieldValueError: If the value is not between 1 and 8
        """
        if not (1 <= value <= 8):
            raise FieldValueError(
                "tensor_parallel_size must be between 1 and 8",
                field="tensor_parallel_size",
                value=value
            )

    @staticmethod
    def validate_entity_linking(config: "BaseMedicalConfig") -> None:
        """Validate entity linking configuration.
        
        Args:
            config: The configuration object to validate
            
        Raises:
            ValidationError: If entity linking is enabled but no knowledge bases are specified
        """
        if config.entity_linking.get(
            "enabled", False
        ) and not config.entity_linking.get("knowledge_bases"):
            raise ValidationError(
                "Entity linking is enabled but no knowledge bases are specified",
                field="entity_linking.knowledge_bases"
            )

    @classmethod
    def validate_medical_parameters(cls, config: "BaseMedicalConfig") -> None:
        """Validate all medical-specific parameters.
        
        Args:
            config: The configuration object to validate
            
        Raises:
            ValidationError: If any validation fails
        """
        if (
            hasattr(config, "tensor_parallel_size")
            and config.tensor_parallel_size is not None
        ):
            cls.validate_tensor_parallel_size(config.tensor_parallel_size)

        if hasattr(config, "entity_linking") and config.entity_linking.get("enabled", False):
            cls.validate_entity_linking(config)

    @staticmethod
    def warn_deprecated(param_name: str, version: str, alternative: str = None) -> None:
        """Log a deprecation warning for a parameter.
        
        Args:
            param_name: Name of the deprecated parameter
            version: Version in which the parameter will be removed
            alternative: Alternative parameter or approach to use (optional)
        """
        msg = f"'{param_name}' is deprecated and will be removed in version {version}."
        if alternative:
            msg += f" Use '{alternative}' instead."
        warnings.warn(msg, DeprecationWarning, stacklevel=3)


# Create a default validator instance for convenience
default_validator = MedicalConfigValidator()
