"""
Base serializer for medical model configurations.

This module provides the base serializer class for converting between
configuration objects and their serialized representations.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

# Import the base configuration class
from ..base import BaseMedicalConfig

# Type variable for generic configuration types
T = TypeVar("T", bound=BaseMedicalConfig)


class ConfigSerializer:
    """Base class for configuration serializers.

    This class provides the foundation for serializing and deserializing
    configuration objects to/from various formats (JSON, YAML, etc.).
    """

    @classmethod
    def to_dict(cls, config: Any) -> Dict[str, Any]:
        """Convert a configuration object to a dictionary.

        Args:
            config: The configuration object to serialize (can be a dict,
                   BaseMedicalConfig, or any object with to_dict())

        Returns:
            A dictionary representation of the configuration

        Raises:
            TypeError: If the input cannot be converted to a dictionary
        """
        if isinstance(config, dict):
            return dict(config)

        if hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
            result = config.to_dict()
            if not isinstance(result, dict):
                raise TypeError(
                    "to_dict() did not return a dictionary, "
                    f"got {type(result).__name__}"
                )
            return result

        # Check for BaseMedicalConfig using a safer approach
        if (
            hasattr(config, "__class__")
            and hasattr(config.__class__, "__mro__")
            and any(
                cls.__name__ == "BaseMedicalConfig" for cls in config.__class__.__mro__
            )
        ):
            # Handle BaseMedicalConfig specifically
            output: Dict[str, Any] = {}
            # Get base config parameters from parent classes
            base = super(config.__class__, config)
            if hasattr(base, "to_dict") and callable(base.to_dict):
                base_dict = base.to_dict()
                if isinstance(base_dict, dict):
                    output.update(base_dict)

            # Add medical-specific fields
            medical_fields = [
                "config_version",
                "model",
                "model_type",
                "medical_specialties",
                "anatomical_regions",
                "imaging_modalities",
                "clinical_metrics",
                "regulatory_compliance",
                "use_crf",
                "do_lower_case",
                "preserve_case_for_abbreviations",
            ]

            for field in medical_fields:
                if hasattr(config, field):
                    value = getattr(config, field)
                    if value is not None:
                        output[field] = cls._convert_to_serializable(value)

            return output

        msg = f"Cannot convert object of type {type(config)} to dictionary"
        raise TypeError(msg)

    @classmethod
    def _convert_to_serializable(
        cls, obj: Any
    ) -> Union[dict[str, Any], list[Any], str, int, float, bool, None]:
        """Recursively convert an object to a serializable dictionary.

        Args:
            obj: The object to convert (dict, list, tuple, set, etc.)

        Returns:
            A serializable representation of the object
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, dict):
            return {str(k): cls._convert_to_serializable(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [cls._convert_to_serializable(item) for item in obj]

        # Handle objects with __dict__ attribute
        if hasattr(obj, "__dict__"):
            obj_dict = vars(obj)
            return {
                str(k): cls._convert_to_serializable(v) for k, v in obj_dict.items()
            }

        # Handle numpy types (common in ML models)
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                item_val = obj.item()
                if isinstance(item_val, (str, int, float, bool)):
                    return item_val
                return str(item_val)
            except (TypeError, ValueError):
                pass

        # Handle objects with to_dict method
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                result = obj.to_dict()
                return cls._convert_to_serializable(result)
            except (TypeError, ValueError, AttributeError) as e:
                logger.debug(
                    "Failed to convert object to dict using to_dict(): %s",
                    str(e),
                    exc_info=True,
                )

        # Handle Pydantic models
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            try:
                result = obj.model_dump()
                return cls._convert_to_serializable(result)
            except (TypeError, ValueError, AttributeError) as e:
                logger.debug(
                    "Failed to convert Pydantic model to dict: %s",
                    str(e),
                    exc_info=True,
                )

        # For any other type, convert to string
        return str(obj)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], config_class: Type[T]) -> T:
        """Create a configuration object from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters
            config_class: The configuration class to instantiate

        Returns:
            An instance of the specified configuration class

        Raises:
            TypeError: If config_class is not a subclass of BaseMedicalConfig
            ValueError: If the configuration data is invalid
        """
        if not (
            isinstance(config_class, type)
            and (
                issubclass(config_class, BaseMedicalConfig)
                if hasattr(config_class, "__mro__")
                else False
            )
        ):
            raise TypeError(
                "config_class must be a subclass of BaseMedicalConfig, "
                f"got {config_class}"
            )

        try:
            # Create a new instance of the config class
            return config_class(**config_dict)
        except Exception as e:
            raise ValueError(
                f"Failed to create configuration from dictionary: {e}"
            ) from e

    @classmethod
    def to_json(cls, config: Union[T, Dict[str, Any]], **kwargs: Any) -> str:
        """Convert configuration to a JSON string.

        Args:
            config: The configuration to serialize (can be a config
                   object or dict)
            **kwargs: Additional arguments for json.dumps()

        Returns:
            A JSON string representation of the configuration
        """
        config_dict = cls.to_dict(config) if not isinstance(config, dict) else config
        return json.dumps(config_dict, **kwargs)

    @classmethod
    def from_json(cls, json_str: str, config_class: Type[T], **kwargs: Any) -> T:
        """Create a configuration from a JSON string.

        Args:
            json_str: JSON string containing the configuration
            config_class: The configuration class to instantiate
            **kwargs: Additional arguments for json.loads()

        Returns:
            An instance of the specified configuration class
        """
        config_dict = json.loads(json_str, **kwargs)
        return cls.from_dict(config_dict, config_class)

    @classmethod
    def save_to_file(
        cls,
        config: T,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> None:
        """Save configuration to a file in JSON format.

        Args:
            config: The configuration to save
            file_path: Path to the output file
            encoding: File encoding to use
            **kwargs: Additional arguments for json.dump()
        """
        config_dict = cls.to_dict(config)
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(config_dict, f, **kwargs)

    @classmethod
    def load_from_file(
        cls,
        file_path: Union[str, Path],
        config_class: Type[T],
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> T:
        """Load configuration from a JSON file.

        Args:
            file_path: Path to the input file
            config_class: The configuration class to instantiate
            encoding: File encoding to use
            **kwargs: Additional arguments for json.load()

        Returns:
            An instance of the specified configuration class
        """
        with open(file_path, "r", encoding=encoding) as f:
            config_dict = json.load(f, **kwargs)

        # Update with any additional keyword arguments
        if kwargs:
            config_dict.update(kwargs)

        return cls.from_dict(config_dict, config_class)

    @classmethod
    def _serialize_to_str(cls, data: Dict[str, Any], **kwargs: Any) -> str:
        """Serialize a dictionary to a string.

        Args:
            data: The dictionary to serialize
            **kwargs: Additional arguments for the serializer

        Returns:
            A string representation of the data

        Raises:
            NotImplementedError: If the method is not implemented by a subclass
        """
        raise NotImplementedError("_serialize_to_str must be implemented by a subclass")

    @classmethod
    def _is_serializable(cls, data: Any) -> bool:
        """Check if the data can be serialized.

        Args:
            data: The data to check

        Returns:
            bool: True if the data can be serialized, False otherwise
        """
        """Check if the data can be serialized.

        Args:
            data: The data to check

        Returns:
            bool: True if the data can be serialized, False otherwise
        """
        # Only allow None, basic types, dicts, objects with to_dict(), or
        # BaseMedicalConfig
        if data is None or isinstance(data, (str, int, float, bool, dict)):
            return True

        # Check for objects with to_dict() method
        if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
            return True

        # Check for BaseMedicalConfig instances
        try:
            from ..base import BaseMedicalConfig

            return isinstance(data, BaseMedicalConfig)
        except ImportError:
            return False

    @classmethod
    def serialize(
        cls,
        data: Any,
        file_path: Union[str, Path, None] = None,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> Union[str, None]:
        """Serialize data to a string or file.

        Args:
            data: The data to serialize (can be a dict, BaseMedicalConfig,
                or any object with to_dict())
            file_path: Optional file path to save the serialized data
            encoding: File encoding to use if saving to file
            **kwargs: Additional arguments for the serializer

        Returns:
            The serialized string if file_path is None, otherwise None

        Raises:
            TypeError: If the input cannot be serialized
            OSError: If there's an error writing to the file
        """
        serialized: str = ""

        # Check if the data is serializable - be very strict about what we
        # accept
        if data is None or isinstance(data, (int, float, bool)):
            serialized = str(data)
        elif isinstance(data, dict):
            serialized = cls._serialize_to_str(data, **kwargs)
        # Handle objects with to_dict() method
        elif hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
            serialized = cls._serialize_to_str(data.to_dict(), **kwargs)
        # Handle BaseMedicalConfig instances
        elif hasattr(data, "model_dump") and callable(getattr(data, "model_dump")):
            serialized = cls._serialize_to_str(data.model_dump(), **kwargs)
        # Handle strings - raise TypeError if it's not a file path
        elif isinstance(data, str):
            if not os.path.exists(data):
                raise TypeError(
                    f"Cannot serialize string that is not a file path: {data}"
                )
            # If it's a file path, read the file
            with open(data, "r", encoding=encoding) as f:
                serialized = f.read()
        else:
            msg = f"Unsupported type for serialization: {type(data)}"
            raise TypeError(msg)

        # Handle file output if file_path is provided
        if file_path is not None:
            try:
                with open(file_path, "w", encoding=encoding) as f:
                    f.write(serialized)
                return None
            except OSError as e:
                raise OSError(f"Failed to write to file {file_path}: {str(e)}")

        return serialized

    @classmethod
    def deserialize(
        cls,
        data: Union[str, bytes, Path],
        config_class: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], T]:
        """Deserialize data from a string, bytes, or file.

        Args:
            data: The data to deserialize (string, bytes, or file path)
            config_class: Optional configuration class to instantiate
            **kwargs: Additional arguments for the deserializer

        Returns:
            The deserialized data (dict or instance of config_class
            if provided)

        Raises:
            ValueError: If the input data cannot be deserialized
            TypeError: If the input type is not supported
            OSError: If there's an error reading from the file
        """
        from typing import cast

        # Helper function to process content from string/bytes
        def process_content(content: str) -> Union[Dict[str, Any], T]:
            """Helper to process content and return appropriate type."""
            # Try to parse the content
            try:
                # Try JSON first
                config_dict: Dict[str, Any] = json.loads(content, **kwargs)
            except json.JSONDecodeError:
                # Fall back to YAML if JSON parsing fails
                try:
                    from .yaml_serializer import YAMLSerializer

                    yaml_content = YAMLSerializer.from_yaml(content, **kwargs)
                    if not isinstance(yaml_content, dict):
                        raise ValueError("YAML content did not parse to a dictionary")
                    config_dict = yaml_content
                except (ImportError, ValueError) as e:
                    raise ValueError("Could not parse input as JSON or YAML") from e

            # Return the appropriate type based on config_class
            if config_class is not None:
                return cls.from_dict(config_dict, config_class)
            return config_dict

        # Handle concrete implementation's _deserialize_from_str
        if hasattr(cls, "_deserialize_from_str"):
            if isinstance(data, (str, Path)):
                try:
                    with open(str(data), "r", encoding="utf-8") as f:
                        file_content = f.read()
                    result = cls._deserialize_from_str(file_content, **kwargs)
                    if config_class is not None:
                        if not isinstance(result, config_class):
                            result = cls.from_dict(result, config_class)
                        return cast(T, result)
                    return cast(Dict[str, Any], result)
                except OSError as e:
                    # Raise with a simple error message to match test
                    # expectations
                    raise OSError("Error reading file") from e
            elif isinstance(data, bytes):
                file_content = data.decode("utf-8")
                result = cls._deserialize_from_str(file_content, **kwargs)
                if config_class is not None:
                    if not isinstance(result, config_class):
                        result = cls.from_dict(result, config_class)
                    return cast(T, result)
                return cast(Dict[str, Any], result)
            else:
                raise TypeError(
                    f"Unsupported data type for deserialization: {type(data)}"
                )

        # Handle standard deserialization
        if isinstance(data, (str, Path)):
            try:
                with open(str(data), "r", encoding="utf-8") as f:
                    content = f.read()
                result = process_content(content)
                return cast(Union[Dict[str, Any], T], result)
            except OSError as e:
                # Raise with a simple error message to match test expectations
                raise OSError("Error reading file") from e
        elif isinstance(data, bytes):
            content = data.decode("utf-8")
            result = process_content(content)
            return cast(Union[Dict[str, Any], T], result)

        msg = f"Unsupported data type for deserialization: {type(data)}"
        raise TypeError(msg)
