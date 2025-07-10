"""
Schema validation for medical model configurations.

This module provides functions for validating configuration schemas.
"""

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo

# Type variable for Pydantic models
ModelT = TypeVar("ModelT", bound=BaseModel)

from medvllm.medical.config.validation.exceptions import (
    FieldTypeError,
    FieldValueError,
    RequiredFieldError,
    SchemaValidationError,
)


def validate_config_schema(config: Any, config_class: Type[ModelT]) -> ModelT:
    """Validate a configuration object against its schema.

    Args:
        config: Configuration dictionary or object to validate
        config_class: Pydantic model class to validate against

    Returns:
        Validated configuration object

    Raises:
        SchemaValidationError: If validation fails
    """
    if config is None:
        raise SchemaValidationError("Configuration cannot be None")

    # Convert to dict if it's a model instance
    if isinstance(config, BaseModel):
        config = config.dict()
    elif not isinstance(config, dict):
        raise SchemaValidationError(
            f"Expected dict or BaseModel, got {type(config).__name__}"
        )

    try:
        # Check for missing required fields
        required_fields = get_required_fields(config_class)
        missing = [field for field, _ in required_fields.items() if field not in config]
        if missing:
            raise RequiredFieldError(
                f"Missing required fields: {', '.join(missing)}",
                field=missing[0] if len(missing) == 1 else None,
            )

        # Get model fields
        model_fields = get_model_fields(config_class)

        # Validate field types before Pydantic validation
        for field_name, field_value in config.items():
            if field_name not in model_fields:
                continue

            field = model_fields[field_name]

            # Skip None values for optional fields
            if field_value is None and not is_field_required(field):
                continue

            # Get the field type annotation
            field_type = get_field_type(field)

            # Skip if we can't determine the type
            if field_type is None or field_type == inspect.Parameter.empty:
                continue

            # Handle nested models
            if (
                inspect.isclass(field_type)
                and issubclass(field_type, BaseModel)
                and isinstance(field_value, dict)
            ):
                # Recursively validate nested models
                validate_config_schema(field_value, field_type)
                continue

            # Handle lists of models
            origin = get_origin(field_type)
            if origin is not None and issubclass(origin, list):
                args = get_args(field_type)
                if (
                    args
                    and len(args) == 1
                    and inspect.isclass(args[0])
                    and issubclass(args[0], BaseModel)
                    and isinstance(field_value, list)
                ):
                    for item in field_value:
                        if isinstance(item, dict):
                            validate_config_schema(item, args[0])

        # Validate using Pydantic
        return config_class(**config)

    except PydanticValidationError as e:
        # Convert Pydantic validation errors to our custom exceptions
        for error in e.errors():
            error_type = error["type"]
            msg = error["msg"]
            field = ".".join(str(loc) for loc in error["loc"])

            if error_type == "value_error.missing":
                raise RequiredFieldError(
                    f"Missing required field: {field}", field=field
                ) from e
            elif error_type.startswith("type_error"):
                raise FieldTypeError(
                    f"Invalid type for field '{field}': {msg}",
                    field=field,
                    value=error.get("input"),
                ) from e
            elif error_type.startswith("value_error"):
                raise FieldValueError(
                    f"Invalid value for field '{field}': {msg}",
                    field=field,
                    value=error.get("input"),
                ) from e

        # Validate field values
        if "age" in config and config["age"] is not None and config["age"] < 0:
            raise FieldValueError(
                "value is less than 0", field="age", value=config["age"]
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
                        f"Missing required field: {field}", field=field
                    ) from e
                elif error_type.startswith("type_error"):
                    raise FieldTypeError(
                        f"Invalid type for field '{field}': {msg}",
                        field=field,
                        value=error.get("input"),
                    ) from e
                elif error_type.startswith("value_error"):
                    raise FieldValueError(
                        f"Invalid value for field '{field}': {msg}",
                        field=field,
                        value=error.get("input"),
                    ) from e

            raise SchemaValidationError(
                f"Configuration validation failed: {str(e)}"
            ) from e

    return config


def is_field_required(field: Any) -> bool:
    """Check if a field is required in a version-agnostic way."""
    if hasattr(field, "is_required"):  # Pydantic v2
        return field.is_required()
    return getattr(field, "required", False)  # type: ignore  # Pydantic v1


def get_field_type(field: Any) -> Optional[type]:
    """Get the type of a field in a version-agnostic way."""
    if hasattr(field, "annotation"):  # Pydantic v2
        return field.annotation
    return getattr(field, "outer_type_", None)  # type: ignore  # Pydantic v1


def get_model_fields(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Get model fields in a version-agnostic way."""
    if hasattr(model_class, "model_fields"):  # Pydantic v2
        return model_class.model_fields  # type: ignore
    elif hasattr(model_class, "__fields__"):  # Pydantic v1
        return model_class.__fields__  # type: ignore
    return {}


def get_required_fields(config_class: Type[ModelT]) -> Dict[str, type]:
    """Get a dictionary of required fields and their types for a config class.

    Args:
        config_class: Pydantic model class

    Returns:
        Dictionary mapping field names to their types
    """
    type_hints = get_type_hints(config_class, include_extras=True)
    model_fields = get_model_fields(config_class)
    result: Dict[str, type] = {}

    for field_name, field in model_fields.items():
        if not is_field_required(field):
            continue

        # Get field type from annotation or type hints
        field_type = get_field_type(field)
        if field_type is None or field_type == inspect.Parameter.empty:
            field_type = type_hints.get(field_name, Any)

        # Handle Optional and Union types
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            if type(None) in args:  # Skip Optional fields
                continue

        result[field_name] = field_type

    return result
