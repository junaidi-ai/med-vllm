"""
JSON serialization for medical model configurations.

This module provides JSON serialization and deserialization for medical
model configurations, with support for both file and string I/O.

This module uses Python's built-in json module and does not require any
external dependencies for basic functionality.
"""

import dataclasses
import json
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

# Import the base configuration class
from ..base import BaseMedicalConfig
from .config_serializer import ConfigSerializer

# Set up logger
logger = logging.getLogger(__name__)

# Import the model configuration class with proper type checking
if TYPE_CHECKING:  # pragma: no cover
    pass

# Type variable for generic configuration types
T = TypeVar("T", bound=BaseMedicalConfig)

# Module-level exports
__all__ = ["JSONSerializer"]


class JSONSerializer(ConfigSerializer):
    """JSON serializer for medical model configurations.

    This class provides methods for serializing and deserializing
    configuration objects to/from JSON format, with support for both
    file and string I/O.
    """

    @classmethod
    def to_json(
        cls,
        config: Union[T, Dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Convert configuration to a JSON string.

        Args:
            config: The configuration to serialize (can be a config
                   object or dict)
            **kwargs: Additional arguments for json.dumps()

        Returns:
            A JSON string representation of the configuration

        Raises:
            TypeError: If the input is not a valid configuration object
            ValueError: If serialization fails
        """
        try:
            # Convert config to dictionary if it's not already one
            if not isinstance(config, dict):
                config_dict = super().to_dict(config)
            else:
                config_dict = config

            # Set default JSON serialization options if not provided
            kwargs.setdefault("indent", 2)
            kwargs.setdefault("ensure_ascii", False)

            # Convert to JSON string and return
            return json.dumps(config_dict, **kwargs)

        except Exception as e:
            raise ValueError(f"Failed to serialize configuration: {e}") from e

    @classmethod
    def from_json(
        cls,
        json_str: str,
        config_class: Type[T],
        **json_kwargs: Any,
    ) -> T:
        """Deserialize configuration from JSON.

        Args:
            json_input: JSON string, bytes, or file path to load from
            config_class: The configuration class to instantiate
            **json_kwargs: Additional arguments to pass to json.loads()

        Returns:
            Deserialized configuration object

        Raises:
            ValueError: If deserialization fails or the configuration
                       is invalid
        """
        from medvllm.medical.config.models.schema import MedicalModelConfigSchema

        try:
            # Parse JSON string to dict
            json_dict = json.loads(json_str, **json_kwargs)

            # Validate against the schema first
            schema_data = MedicalModelConfigSchema.model_validate(
                json_dict
            ).model_dump()

            # Map schema fields to config fields
            config_data = {}
            field_mapping = {"max_sequence_length": "max_medical_seq_length"}

            for schema_field, value in schema_data.items():
                # Map schema field name to config field name if needed
                config_field = field_mapping.get(schema_field, schema_field)
                config_data[config_field] = value

            # Create a new config instance with the mapped data
            config = config_class.__new__(config_class)

            # Set the fields directly on the instance
            for field, value in config_data.items():
                if hasattr(config_class, field):
                    setattr(config, field, value)

            return config

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
        except FileNotFoundError as e:
            raise ValueError(f"File not found: {e}") from e
        except Exception as e:
            # Catch any other exceptions and wrap them in a ValueError
            msg = f"Failed to deserialize configuration: {e}"
            raise ValueError(msg) from e

    @classmethod
    def _convert_dataclass_to_dict(cls, value: Any) -> Union[Dict[str, Any], str]:
        """Convert a dataclass instance to a dictionary.

        Args:
            value: A dataclass instance

        Returns:
            A dictionary representation of the dataclass
            or a string if conversion fails
        """
        data: Dict[str, Any] = {}

        try:
            # Get fields from the dataclass type
            for field in dataclasses.fields(value):
                # Skip private fields
                if field.name.startswith("_"):
                    continue

                # Get and convert field value
                try:
                    field_value = getattr(value, field.name, None)
                    if field_value is not None:
                        data[field.name] = cls._convert_to_serializable(field_value)
                except (AttributeError, TypeError, ValueError) as e:
                    # Log and skip fields that can't be accessed or converted
                    logger.debug(
                        "Skipping field %s due to error: %s",
                        field.name,
                        str(e),
                        exc_info=True,
                    )
                    continue

            return data if data else str(value)
        except Exception:
            # If anything goes wrong, return string representation
            return str(value)

    @classmethod
    def _convert_to_serializable(
        cls, value: Any
    ) -> Union[str, int, float, bool, List[Any], Dict[str, Any], None]:
        """Recursively convert a value to a JSON-serializable format.

        Handles conversion of various Python types to JSON-serializable
        formats. Supported types:
        - Basic types: None, str, int, float, bool
        - Collections: list, tuple, set, dict
        - Enums: converted to their values
        - Dataclasses: converted to dict
        - Pydantic BaseModels: using .dict()
        - Objects: with to_dict() method
        - Other objects: converted to strings

        Args:
            value: The value to convert to a JSON-serializable format

        Returns:
            A JSON-serializable representation of the value
            (str, int, float, bool, list, or dict)

        Raises:
            ValueError: If the value cannot be converted to a
                JSON-serializable format
        """
        # Handle None and basic types first
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        # Handle collections
        if isinstance(value, (list, tuple, set)):
            return [cls._convert_to_serializable(item) for item in value]

        if isinstance(value, dict):
            return {str(k): cls._convert_to_serializable(v) for k, v in value.items()}

        # Handle enums
        if isinstance(value, Enum):
            enum_value = value.value
            # Ensure the enum value is JSON-serializable
            is_basic_type = isinstance(enum_value, (str, int, float, bool))
            if is_basic_type or enum_value is None:
                return enum_value
            return str(enum_value)

        # Handle dataclasses
        is_dataclass = dataclasses.is_dataclass(value)
        is_not_type = not isinstance(value, type)
        is_dataclass = is_dataclass and is_not_type
        if is_dataclass:
            # Process dataclass instance
            return cls._convert_dataclass_to_dict(value)

        # Handle Pydantic models
        if hasattr(value, "dict") and callable(getattr(value, "dict", None)):
            dict_value = value.dict()
            if not isinstance(dict_value, dict):
                return str(dict_value)
            # Convert all values in the dict to be JSON-serializable
            return {
                str(k): cls._convert_to_serializable(v) for k, v in dict_value.items()
            }

        # Handle objects with to_dict method
        has_to_dict = hasattr(value, "to_dict")
        is_callable = callable(getattr(value, "to_dict", None))
        if has_to_dict and is_callable:
            dict_value = value.to_dict()
            if not isinstance(dict_value, (str, int, float, bool, list, dict)):
                return str(dict_value)
            if isinstance(dict_value, dict):
                return {
                    str(k): cls._convert_to_serializable(v)
                    for k, v in dict_value.items()
                }
            return dict_value

        # For any other type, convert to string
        try:
            return str(value)
        except Exception as e:
            raise ValueError(
                f"Could not convert value of type {type(value).__name__} "
                f"to a JSON-serializable format: {e}"
            ) from e
