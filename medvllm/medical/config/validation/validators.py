"""
Validators for medical model configurations.

This module contains various validators for configuration values.
"""

import warnings
from typing import TYPE_CHECKING, Any

from .exceptions import FieldValueError, ValidationError

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
            FieldValueError: If the value is not between 1 and 8 or
                           is not an integer
        """
        if value is None:
            raise FieldValueError(
                "must be between 1 and 8", field="tensor_parallel_size", value=value
            )

        try:
            int_value = int(value)
        except (TypeError, ValueError):
            raise FieldValueError(
                "must be between 1 and 8", field="tensor_parallel_size", value=value
            )

        if not (1 <= int_value <= 8):
            raise FieldValueError(
                "must be between 1 and 8", field="tensor_parallel_size", value=value
            )

    @staticmethod
    def validate_entity_linking(config: "BaseMedicalConfig") -> None:
        """Validate entity linking configuration.

        Args:
            config: The configuration object to validate

        Raises:
            ValidationError: If entity linking is enabled but no
                           knowledge bases are specified
        """
        # Use getattr with a default value to safely access entity_linking
        entity_linking = getattr(config, "entity_linking", None)
        if entity_linking is None:
            return

        # If entity_linking is present but not a dictionary, it's invalid
        if not isinstance(entity_linking, dict):
            raise ValidationError(
                "Entity linking configuration must be a dictionary",
                field="entity_linking",
            )

        # Check if entity linking is enabled and has knowledge bases
        if entity_linking.get("enabled", False):
            knowledge_bases = entity_linking.get("knowledge_bases")
            if (
                not knowledge_bases
                or not isinstance(knowledge_bases, list)
                or not knowledge_bases
            ):
                raise ValidationError(
                    "Entity linking is enabled but no knowledge " "bases are specified",
                    field="entity_linking.knowledge_bases",
                )

    @classmethod
    def validate_medical_parameters(cls, config: "BaseMedicalConfig") -> None:
        """Validate all medical-specific parameters.

        Args:
            config: The configuration object to validate

        Raises:
            ValidationError: If any validation fails
        """
        # Use getattr to safely access attributes
        tensor_parallel_size = getattr(config, "tensor_parallel_size", None)
        if tensor_parallel_size is not None:
            cls.validate_tensor_parallel_size(tensor_parallel_size)

        # Check if entity_linking exists and is enabled
        entity_linking = getattr(config, "entity_linking", None)
        if isinstance(entity_linking, dict) and entity_linking.get("enabled", False):
            cls.validate_entity_linking(config)

    @staticmethod
    def warn_deprecated(param_name: str, version: str, alternative: str = "") -> None:
        """Log a deprecation warning for a parameter.

        Args:
            param_name: Name of the deprecated parameter
            version: Version in which the parameter will be removed
            alternative: Alternative parameter or approach to use (optional)
        """
        msg = (
            f"'{param_name}' is deprecated and will be removed in "
            f"version {version}."
        )
        if alternative:
            msg += f" Use '{alternative}' instead."
        warnings.warn(msg, DeprecationWarning, stacklevel=3)


# Create a default validator instance for convenience
default_validator = MedicalConfigValidator()


def validate_config(config: "BaseMedicalConfig") -> None:
    """Validate a configuration using the default validator.
    
    This is a convenience function that uses the default validator instance
    to validate the configuration.
    
    Args:
        config: The configuration to validate
        
    Raises:
        ValidationError: If validation fails
    """
    default_validator.validate(config)


def validate_field(
    config: "BaseMedicalConfig", field_name: str, value: Any
) -> None:
    """Validate a single field in a configuration.
    
    This is a convenience function that validates a single field in a configuration
    using the default validator.
    
    Args:
        config: The configuration containing the field
        field_name: The name of the field to validate
        value: The value to validate
        
    Raises:
        ValidationError: If validation fails
    """
    if not hasattr(default_validator, f"validate_{field_name}"):
        raise ValueError(f"No validator found for field: {field_name}")
        
    validator = getattr(default_validator, f"validate_{field_name}")
    validator(config, value)
