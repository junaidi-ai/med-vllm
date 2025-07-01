"""Validation utilities for medical model configurations.

This package provides a comprehensive validation framework for medical
model configurations, including schema validation, type checking, and
custom validation rules.

Key components:
- `MedicalConfigValidator`: Main validator class with built-in
  validation methods
- `ValidationError`: Base exception for validation errors
- `validate_config_schema`: Function for schema validation
- Various specialized exceptions for different validation scenarios
"""

from .exceptions import (
    FieldTypeError,
    FieldValueError,
    RequiredFieldError,
    SchemaValidationError,
    ValidationError,
    VersionCompatibilityError,
)
from .schema import get_required_fields, validate_config_schema
from .validators import MedicalConfigValidator, default_validator

__all__ = [
    # Validators
    "MedicalConfigValidator",
    "default_validator",
    # Exceptions
    "ValidationError",
    "SchemaValidationError",
    "VersionCompatibilityError",
    "RequiredFieldError",
    "FieldTypeError",
    "FieldValueError",
    # Schema utilities
    "validate_config_schema",
    "get_required_fields",
]
