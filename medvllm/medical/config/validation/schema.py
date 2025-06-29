"""
Schema validation for medical model configurations.

This module provides functions for validating configuration schemas.
"""

import inspect
from typing import Any, Dict, Type, get_type_hints, get_origin, get_args, Union, Optional
from pydantic import BaseModel, ValidationError as PydanticValidationError

from medvllm.medical.config.validation.exceptions import (
    SchemaValidationError,
    FieldTypeError,
    FieldValueError,
    RequiredFieldError
)

def validate_config_schema(config: Any, config_class: Type[BaseModel]) -> Any:
    """Validate a configuration object against its schema.
    
    Args:
        config: Configuration dictionary or object to validate
        config_class: Pydantic model class to validate against
        
    Returns:
        Validated configuration object
        
    Raises:
        SchemaValidationError: If validation fails
    """
    if not isinstance(config, config_class):
        if not isinstance(config, dict):
            config = config.dict() if hasattr(config, 'dict') else dict(config)
            
        # First check for missing required fields
        required_fields = get_required_fields(config_class)
        missing = [field for field in required_fields if field not in config]
        if missing:
            raise RequiredFieldError(
                f"Missing required field(s): {', '.join(missing)}",
                field=missing[0] if len(missing) == 1 else None
            )
        
        # Get model fields
        model_fields = config_class.model_fields if hasattr(config_class, 'model_fields') else config_class.__fields__
        
        # Validate field types before Pydantic validation
        for field_name, field_value in config.items():
            if field_name not in model_fields:
                continue
                
            field = model_fields[field_name]
            
            # Skip None values for optional fields
            if field_value is None and not field.is_required():
                continue
                
            # Get the field type annotation
            field_type = field.annotation if hasattr(field, 'annotation') else field.outer_type_
            
            # Skip if we can't determine the type
            if field_type is None or field_type == inspect.Parameter.empty:
                continue
                
            # Handle nested models
            if hasattr(field_type, '__base__') and issubclass(field_type, BaseModel):
                if field_value is not None:
                    if isinstance(field_value, dict):
                        # Recursively validate nested model
                        config[field_name] = validate_config_schema(field_value, field_type)
                continue
                
            # Handle subscripted generic types (List, Dict, etc.)
            origin = get_origin(field_type)
            if origin is not None:
                # For List types, verify the input is iterable
                if origin == list:
                    if not isinstance(field_value, (list, tuple, set)):
                        raise FieldTypeError(
                            "invalid type",
                            field=field_name,
                            value=field_value
                        )
                    # Convert to list if it's a set or tuple
                    if not isinstance(field_value, list):
                        config[field_name] = list(field_value)
                # For other generic types, skip direct type checking
                # as we can't do isinstance() with subscripted generics in Python
                continue
                
            # Check type
            try:
                # Handle boolean validation
                if field_type == bool:
                    if not isinstance(field_value, bool):
                        raise FieldTypeError(
                            "invalid type",
                            field=field_name,
                            value=field_value
                        )
                # Only try conversion if types don't match and it's not a generic type
                elif not isinstance(field_value, field_type):
                    try:
                        config[field_name] = field_type(field_value)
                    except (ValueError, TypeError) as e:
                        raise FieldTypeError(
                            f"Invalid type for field '{field_name}': {str(e)}",
                            field=field_name,
                            value=field_value
                        )
            except (ValueError, TypeError) as e:
                raise FieldTypeError(
                    f"Invalid type for field '{field_name}': {str(e)}",
                    field=field_name,
                    value=field_value
                )
        
        # Validate field values
        if 'age' in config and config['age'] is not None and config['age'] < 0:
            raise FieldValueError(
                "value is less than 0",
                field='age',
                value=config['age']
            )
        
        # Create the model instance
        try:
            config = config_class(**config)
        except PydanticValidationError as e:
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                error_type = error["type"]
                
                if error_type == "value_error.missing":
                    raise RequiredFieldError(
                        f"Missing required field: {field}",
                        field=field
                    ) from e
                elif error_type.startswith("type_error"):
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
                
            raise SchemaValidationError(
                f"Configuration validation failed: {str(e)}"
            ) from e
            
    return config


def get_required_fields(config_class: Type[BaseModel]) -> Dict[str, type]:
    """Get a dictionary of required fields and their types for a config class.
    
    Args:
        config_class: Pydantic model class
        
    Returns:
        Dictionary mapping field names to their types
    """
    from typing import get_origin, get_args, Union, Optional
    
    # Get type hints and model fields based on Pydantic version
    type_hints = get_type_hints(config_class, include_extras=True)
    model_fields = config_class.model_fields if hasattr(config_class, 'model_fields') else config_class.__fields__
    
    required_fields = {}
    
    for field_name, field in model_fields.items():
        # Check if field is required based on Pydantic version
        is_required = field.is_required() if hasattr(field, 'is_required') else field.required
        
        if is_required:
            # Get field type from annotation or type_hints
            field_type = getattr(field, 'annotation', None)
            if field_type is None or field_type == inspect.Parameter.empty:
                field_type = type_hints.get(field_name, Any)
            
            # Handle Optional and Union types
            origin = get_origin(field_type)
            if origin is Union:
                # For Optional[Type], extract the actual type
                args = get_args(field_type)
                if type(None) in args:
                    # This is an Optional type, skip it as it's not required
                    continue
                    
            required_fields[field_name] = field_type
    
    return required_fields
