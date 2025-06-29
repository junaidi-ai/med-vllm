"""
JSON serialization for medical model configurations.

This module provides JSON serialization and deserialization for medical
model configurations, with support for both file and string I/O.

This module uses Python's built-in json module and does not require any
external dependencies for basic functionality.
"""

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel

# Import the base configuration class
from ..base import BaseMedicalConfig
from .config_serializer import ConfigSerializer

# Import the model configuration class with proper type checking
if TYPE_CHECKING:
    from ..models.medical_config import MedicalModelConfig

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
    @overload
    def to_json(
        cls,
        config: T,
        file_path: None = None,
        **json_kwargs: Any,
    ) -> str: ...

    @classmethod
    @overload
    def to_json(
        cls,
        config: T,
        file_path: Union[str, Path],
        **json_kwargs: Any,
    ) -> None: ...

    @classmethod
    def to_json(
        cls,
        config: T,
        file_path: Optional[Union[str, Path]] = None,
        **json_kwargs: Any,
    ) -> Optional[str]:
        """Serialize configuration to JSON.

        Args:
            config: Configuration to serialize
            file_path: Optional file path to save the JSON to. If None,
                     returns the JSON as a string.
            **json_kwargs: Additional arguments to pass to json.dump()

        Returns:
            JSON string if file_path is None, else None

        Raises:
            TypeError: If the input is not a valid configuration object
            ValueError: If serialization fails
        """
        from medvllm.medical.config.models.schema import (
            MedicalModelConfigSchema,
            ModelType,
        )

        try:
            # Create a new schema instance with only the allowed fields
            schema_data = {}

            # Map config fields to schema fields
            field_mapping = {"max_medical_seq_length": "max_sequence_length"}

            # Get all fields from the schema
            schema_fields = MedicalModelConfigSchema.model_fields

            # Only include fields that are in the schema
            for field_name in schema_fields:
                # Map schema field name to config field name if needed
                config_field = next(
                    (k for k, v in field_mapping.items() if v == field_name), field_name
                )

                # Get the value from the config if it exists
                if hasattr(config, config_field):
                    value = getattr(config, config_field)
                    # Handle enum values
                    if hasattr(value, "value"):
                        value = value.value
                    schema_data[field_name] = value

            # Handle required fields with defaults if not set
            if "model" not in schema_data and hasattr(config, "model"):
                schema_data["model"] = config.model

            if "model_type" not in schema_data and hasattr(config, "model_type"):
                model_type = config.model_type
                if hasattr(model_type, "value"):
                    schema_data["model_type"] = model_type.value
                else:
                    schema_data["model_type"] = str(model_type).lower()

            # Set default values for optional fields if they're in the schema but not in the config
            if "learning_rate" not in schema_data and "learning_rate" in schema_fields:
                schema_data["learning_rate"] = 5e-5

            if (
                "num_train_epochs" not in schema_data
                and "num_train_epochs" in schema_fields
            ):
                schema_data["num_train_epochs"] = 3

            # Create a new schema instance with just the allowed fields
            schema_instance = MedicalModelConfigSchema(**schema_data)

            # Convert the schema instance to a dictionary first
            config_dict = schema_instance.model_dump()

            # Set default JSON serialization options if not provided
            json_kwargs.setdefault("indent", 2)
            json_kwargs.setdefault("ensure_ascii", False)

            # Convert to JSON string
            json_str = json.dumps(config_dict, **json_kwargs)

            # Write to file if path is provided
            if file_path is not None:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(json_str)
                except IOError as e:
                    raise ValueError(f"Failed to write to file {file_path}: {e}") from e
                return None

            # Otherwise return as JSON string
            return json_str

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
            ValueError: If deserialization fails or the configuration is invalid
        """
        from medvllm.medical.config.models.medical_config import MedicalModelConfig
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
            raise ValueError(f"Failed to deserialize configuration: {e}") from e

    @classmethod
    def _convert_dataclass_to_dict(cls, value: Any) -> Union[Dict[str, Any], str]:
        """Convert a dataclass instance to a dictionary.

        Args:
            value: A dataclass instance

        Returns:
            A dictionary representation of the dataclass or a string if conversion fails
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
                except Exception:
                    # Skip fields that can't be accessed
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

        This method handles conversion of various Python types to JSON-serializable
        formats, including:
        - Basic types (None, str, int, float, bool)
        - Collections (list, tuple, set, dict)
        - Enums (converted to their values)
        - Dataclasses (converted to dict)
        - Pydantic BaseModels (using .dict())
        - Objects with to_dict() method
        - All other objects are converted to strings

        Args:
            value: The value to convert to a JSON-serializable format

        Returns:
            A JSON-serializable representation of the value (str, int, float, bool, list, or dict)

        Raises:
            ValueError: If the value cannot be converted to a JSON-serializable format
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
            if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
                return enum_value
            return str(enum_value)

        # Handle dataclasses
        is_dataclass = dataclasses.is_dataclass(value) and not isinstance(value, type)
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
        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict", None)):
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
                f"Could not convert value of type {type(value).__name__} to a JSON-serializable format: {e}"
            ) from e
