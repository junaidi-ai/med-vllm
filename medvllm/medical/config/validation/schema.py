"""
Schema validation for medical model configurations.

This module provides functions for validating configuration schemas.
"""

from typing import Any, Dict, Type, get_type_hints

from pydantic import BaseModel, ValidationError as PydanticValidationError

from .exceptions import SchemaValidationError, FieldTypeError, FieldValueError


def validate_config_schema(config: Any, config_class: Type[BaseModel]) -> None:
    """Validate a configuration object against its schema.
    
    Args:
        config: Configuration dictionary or object to validate
        config_class: Pydantic model class to validate against
        
    Raises:
        SchemaValidationError: If validation fails
    """
    try:
        if not isinstance(config, config_class):
            config = config_class(**config)
        return config
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            error_type = error["type"]
            
            if error_type.startswith("type_error"):
                raise FieldTypeError(
                    f"Invalid type for field '{field}': {msg}",
                    field=field,
                    value=error.get("input")
                ) from e
            elif error_type.startswith("value_error"):
                raise FieldValueError(
                    f"Invalid value for field '{field}': {msg}",
                    field=field,
                    value=error.get("input")
                ) from e
            else:
                errors.append(f"{field}: {msg}")
        
        if errors:
            raise SchemaValidationError(
                f"Configuration validation failed: {', '.join(errors)}"
            ) from e


def get_required_fields(config_class: Type[BaseModel]) -> Dict[str, type]:
    """Get a dictionary of required fields and their types for a config class.
    
    Args:
        config_class: Pydantic model class
        
    Returns:
        Dictionary mapping field names to their types
    """
    type_hints = get_type_hints(config_class)
    required_fields = {}
    
    for field_name, field in config_class.__fields__.items():
        if field.required:
            required_fields[field_name] = type_hints.get(field_name, Any)
    
    return required_fields
