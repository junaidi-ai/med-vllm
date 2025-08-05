"""Validation utilities for medical model configurations.

This package provides a comprehensive validation framework for medical
model configurations, including schema validation, type checking, and
custom validation rules.
"""

# Import exceptions
from .exceptions import (
    ConfigValidationError,
    FieldTypeError,
    FieldValidationError,
    FieldValueError,
    RequiredFieldError,
    SchemaValidationError,
    ValidationError,
    VersionCompatibilityError,
)

# Import schema validation
from .schema import (
    get_required_fields,
    validate_config_schema,
    validate_schema,
)

# Import validators
from .validators import (
    MedicalConfigValidator,
    default_validator,
    validate_config,
    validate_field,
)

# Re-export everything that should be available when importing from validation
__all__ = [
    # Validators
    "MedicalConfigValidator",
    "default_validator",
    "validate_config",
    "validate_field",
    "validate_schema",
    "validate_config_schema",
    # Exceptions
    "ValidationError",
    "ConfigValidationError",
    "SchemaValidationError",
    "FieldValidationError",
    "FieldTypeError",
    "FieldValueError",
    "RequiredFieldError",
    "VersionCompatibilityError",
    # Schema utilities
    "get_required_fields",
]
