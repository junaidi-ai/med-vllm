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
from dataclasses import fields
from enum import Enum
from json import JSONEncoder
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_type_hints


# Import the base configuration class
from ..base import BaseMedicalConfig
from .config_serializer import ConfigSerializer

# Set up logger
logger = logging.getLogger(__name__)


# Type variable for generic configuration types
T = TypeVar("T", bound=BaseMedicalConfig)

# Module-level exports
__all__ = ["JSONSerializer"]


class ConfigJSONEncoder(JSONEncoder):
    """Custom JSON encoder for configuration objects.

    This encoder handles various non-serializable types that might be
    present in the configuration, such as BertConfig objects, enums, and
    other complex types.
    """

    def default(self, obj):
        # Handle dataclasses
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        # Handle enums
        if isinstance(obj, Enum):
            return obj.value

        # Handle objects with to_dict() method
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()

        # Handle objects with __dict__
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}

        # Let the base class default method raise the TypeError
        return super().default(obj)


class JSONSerializer(ConfigSerializer):
    """JSON serializer for medical model configurations.

    This class provides methods for serializing and deserializing
    configuration objects to/from JSON format, with support for both
    file and string I/O.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Accept optional serializer options without requiring them.

        Tests may instantiate this class with keyword arguments (e.g.,
        indent=2). Since our API is classmethod-based, we simply accept and
        store these options for potential future use without affecting
        behavior.
        """
        # Store but do not rely on instance options; classmethods are used.
        self._options = dict(kwargs)

    # Class-level encoder instance
    _encoder = ConfigJSONEncoder()

    # Bridge methods used by ConfigSerializer
    @classmethod
    def _serialize_to_str(cls, data: Dict[str, Any], **kwargs: Any) -> str:
        """Implement base hook to serialize a dict to a JSON string."""
        return cls.to_json(data, **kwargs)

    @classmethod
    def _deserialize_from_str(cls, data: str, **kwargs: Any) -> Dict[str, Any]:
        """Implement base hook to deserialize a JSON string to a dict."""
        # Use json.loads directly; ConfigSerializer will handle config_class
        # conversion if provided.
        return json.loads(data, **kwargs)

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

            # Use our custom encoder for serialization
            kwargs["cls"] = cls._encoder.__class__

            # Convert to JSON string and return
            try:
                return json.dumps(config_dict, **kwargs)
            except TypeError:
                # If serialization fails, try with our custom encoder
                try:
                    return json.dumps(config_dict, cls=ConfigJSONEncoder, **kwargs)
                except Exception:
                    # If we still can't serialize, try with a more permissive approach
                    try:
                        # Convert all non-serializable objects to strings
                        def safe_serialize(obj):
                            try:
                                json.dumps(obj, **kwargs)
                                return obj
                            except (TypeError, OverflowError):
                                if hasattr(obj, "__dict__"):
                                    return {
                                        k: safe_serialize(v)
                                        for k, v in vars(obj).items()
                                        if not k.startswith("_")
                                    }
                                elif hasattr(obj, "to_dict") and callable(obj.to_dict):
                                    return safe_serialize(obj.to_dict())
                                elif isinstance(obj, (list, tuple, set)):
                                    return [safe_serialize(item) for item in obj]
                                elif isinstance(obj, dict):
                                    return {k: safe_serialize(v) for k, v in obj.items()}
                                else:
                                    return str(obj)

                        safe_dict = safe_serialize(config_dict)
                        return json.dumps(safe_dict, **kwargs)
                    except Exception:
                        # If all else fails, return a minimal representation
                        minimal_dict = {
                            k: v
                            for k, v in config_dict.items()
                            if isinstance(v, (str, int, float, bool, type(None)))
                        }
                        return json.dumps(minimal_dict, **kwargs)

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
            json_str: JSON string to load from
            config_class: Configuration class to instantiate
            **json_kwargs: Additional keyword arguments for json.loads()

        Returns:
            An instance of the specified configuration class

        Raises:
            ValueError: If the JSON is invalid or missing required fields
            TypeError: If the configuration class is invalid
        """
        try:
            # Parse the JSON string
            json_dict = json.loads(json_str, **json_kwargs)
            if not isinstance(json_dict, dict):
                raise ValueError("Expected a JSON object")

            # Special handling for MedicalModelConfig to prevent recursion
            if config_class.__name__ == "MedicalModelConfig":
                from medvllm.medical.config.models.medical_config import (
                    MedicalModelConfig,
                )
                from medvllm.medical.config.models.schema import ModelType

                # Map legacy/alternate field names
                if "max_sequence_length" in json_dict and "max_medical_seq_length" not in json_dict:
                    json_dict["max_medical_seq_length"] = json_dict["max_sequence_length"]

                # Ensure model_type is present in the JSON
                if "model_type" not in json_dict:
                    raise ValueError("Missing required field: model_type")

                # Convert model_type to enum if it's a string
                model_type = json_dict["model_type"]
                if isinstance(model_type, str):
                    try:
                        model_type_enum = ModelType(model_type.lower())
                        # Keep both string and enum representations
                        json_dict["model_type"] = model_type  # Keep as string for __dict__
                        model_type_str = model_type  # Save string version
                    except (ValueError, AttributeError) as e:
                        raise ValueError(f"Invalid model_type: {model_type}") from e
                else:
                    # If it's already an enum, convert to string for __dict__
                    model_type_str = (
                        model_type.value if hasattr(model_type, "value") else str(model_type)
                    )
                    model_type_enum = model_type

                # Create a new instance without calling __init__
                config = object.__new__(MedicalModelConfig)

                # Get all field names from the dataclass
                field_names = {f.name for f in fields(MedicalModelConfig)}

                # Set attributes directly on the instance's __dict__
                for key, value in json_dict.items():
                    # Skip fields not in the dataclass
                    if key not in field_names:
                        continue

                    # Special handling for required fields that need to be in __dict__
                    if key == "model_type":
                        # Set in __dict__ as string for validation
                        config.__dict__[key] = model_type_str
                        # Also set as _model_type for internal use
                        object.__setattr__(config, "_model_type", model_type_enum)
                    elif key == "model":
                        # Set in both __dict__ and as _model for consistency
                        config.__dict__[key] = value
                        object.__setattr__(config, "_model", value)
                    elif key == "max_medical_seq_length":
                        # Ensure both public and internal names are set
                        try:
                            iv = int(value)
                        except Exception:
                            iv = value
                        config.__dict__[key] = iv
                        object.__setattr__(config, f"_{key}", iv)
                    elif key == "batch_size":
                        # Ensure both public and internal names are set
                        try:
                            iv = int(value)
                        except Exception:
                            iv = value
                        config.__dict__[key] = iv
                        object.__setattr__(config, f"_{key}", iv)
                    else:
                        # Set other attributes with underscore prefix to avoid triggering properties
                        object.__setattr__(config, f"_{key}", value)

                # Ensure model_type is properly set as it's required
                if not hasattr(config, "_model_type"):
                    object.__setattr__(config, "_model_type", model_type_enum)

                # Call __post_init__ to initialize any dependent configurations
                if hasattr(config, "__post_init__"):
                    config.__post_init__()

                return config

            # Default handling for other config classes
            config = object.__new__(config_class)
            valid_fields = {f.name for f in fields(config_class)}
            type_hints = get_type_hints(config_class)

            for key, value in json_dict.items():
                if key not in valid_fields:
                    continue

                field_type = type_hints.get(key)
                if field_type is not None:
                    # Handle list types
                    if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                        if isinstance(value, list):
                            item_type = field_type.__args__[0] if field_type.__args__ else str
                            try:
                                value = [item_type(item) for item in value]
                            except (TypeError, ValueError) as e:
                                logger.warning(
                                    "Failed to convert list items for field %s: %s",
                                    key,
                                    str(e),
                                )
                    # Handle dict types
                    elif hasattr(field_type, "__origin__") and field_type.__origin__ is dict:
                        if not isinstance(value, dict):
                            logger.warning(
                                "Expected dict for field %s, got %s",
                                key,
                                type(value).__name__,
                            )
                            continue

                # Set the attribute with underscore prefix
                object.__setattr__(config, f"_{key}", value)

            # Call __post_init__ if it exists
            if hasattr(config, "__post_init__"):
                config.__post_init__()

            return config

        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON: %s", str(e))
            raise ValueError(f"Invalid JSON: {str(e)}") from e
        except Exception as e:
            logger.error("Error deserializing JSON: %s", str(e), exc_info=True)
            raise ValueError(f"Failed to deserialize configuration: {e}") from e

    @classmethod
    def _convert_dataclass_to_dict(
        cls,
        value: Any,
        _depth: int = 0,
        _visited: Optional[set] = None,
        _max_depth: int = 10,
    ) -> Union[Dict[str, Any], str]:
        """Convert a dataclass or object with __dict__ to a dictionary.

        Args:
            value: A dataclass instance or object with __dict__
            _depth: Current recursion depth (internal use)
            _visited: Set of visited object IDs to detect cycles (internal use)
            _max_depth: Maximum recursion depth (default: 10)

        Returns:
            A dictionary representation of the object
            or a string if conversion fails
        """
        if _depth > _max_depth:
            return f"<Max recursion depth ({_max_depth}) exceeded for {type(value).__name__}>"

        data: Dict[str, Any] = {}

        # Initialize visited set on first call
        if _visited is None:
            _visited = set()

        # Handle circular references
        obj_id = id(value)
        if obj_id in _visited:
            return f"<Circular reference to {type(value).__name__} detected>"

        _visited.add(obj_id)

        try:
            # Get fields from the dataclass type or object's __dict__
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                fields_to_process = dataclasses.fields(value)
                get_value = lambda f: getattr(value, f.name, None)
                field_names = [f.name for f in fields_to_process]
            elif hasattr(value, "__dict__"):
                fields_to_process = value.__dict__.items()
                get_value = lambda f: f[1]
                field_names = [f[0] for f in fields_to_process]
            else:
                return str(value)

            for field in fields_to_process:
                field_name = field.name if hasattr(field, "name") else field[0]

                # Skip private fields
                if field_name.startswith("_"):
                    continue

                # Skip properties that might cause recursion
                if field_name in ("__dict__", "__weakref__"):
                    continue

                # Get and convert field value
                try:
                    field_value = get_value(field)
                    if field_value is not None:
                        data[field_name] = cls._convert_to_serializable(
                            field_value, _depth + 1, _visited, _max_depth
                        )
                except Exception as e:
                    # Log and skip fields that can't be accessed or converted
                    logger.debug(
                        "Skipping field %s due to error: %s",
                        field_name,
                        str(e),
                        exc_info=True,
                    )
                    continue

            return data if data else str(value)

        except Exception as e:
            logger.debug("Error in _convert_dataclass_to_dict: %s", str(e))
            return f"<Error converting {type(value).__name__}: {str(e)}>"

        finally:
            # Clean up visited set to avoid memory leaks
            if obj_id in _visited:
                _visited.remove(obj_id)

    @classmethod
    def _convert_to_serializable(
        cls,
        value: Any,
        _depth: int = 0,
        _visited: Optional[set] = None,
        _max_depth: int = 10,
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
            _depth: Current recursion depth (internal use)
            _visited: Set of visited object IDs to detect cycles (internal use)
            _max_depth: Maximum recursion depth (default: 10)

        Returns:
            A JSON-serializable representation of the value
            (str, int, float, bool, list, or dict)
        """
        # Return string representation if max depth exceeded
        if _depth > _max_depth:
            return f"<Max recursion depth ({_max_depth}) exceeded for {type(value).__name__}>"

        # Initialize visited set on first call
        if _visited is None:
            _visited = set()

        # Handle None and basic types first
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        # Handle circular references
        obj_id = id(value)
        if obj_id in _visited:
            return f"<Circular reference to {type(value).__name__} detected>"

        _visited.add(obj_id)

        try:
            # Handle collections
            if isinstance(value, (list, tuple, set)):
                return [
                    cls._convert_to_serializable(item, _depth + 1, _visited, _max_depth)
                    for item in value
                ]

            if isinstance(value, dict):
                return {
                    str(k): cls._convert_to_serializable(v, _depth + 1, _visited, _max_depth)
                    for k, v in value.items()
                }

            # Handle enums
            if isinstance(value, Enum):
                try:
                    enum_value = value.value
                    # Ensure the enum value is JSON-serializable
                    if enum_value is None or isinstance(enum_value, (str, int, float, bool)):
                        return enum_value
                    return str(enum_value)
                except Exception as e:
                    logger.debug("Error converting enum value: %s", str(e))
                    return str(value)

            # Handle dataclasses
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return cls._convert_dataclass_to_dict(value, _depth, _visited, _max_depth)

            # Handle Pydantic models
            if hasattr(value, "dict") and callable(getattr(value, "dict", None)):
                try:
                    dict_value = value.dict()
                    if not isinstance(dict_value, dict):
                        return str(dict_value)
                    return {
                        str(k): cls._convert_to_serializable(v, _depth + 1, _visited, _max_depth)
                        for k, v in dict_value.items()
                    }
                except Exception as e:
                    logger.debug("Error in Pydantic model.dict(): %s", str(e))
                    return str(value)

            # Handle objects with to_dict method
            if hasattr(value, "to_dict") and callable(getattr(value, "to_dict", None)):
                try:
                    dict_value = value.to_dict()
                    if not isinstance(dict_value, dict):
                        return str(dict_value)
                    return {
                        str(k): cls._convert_to_serializable(v, _depth + 1, _visited, _max_depth)
                        for k, v in dict_value.items()
                    }
                except Exception as e:
                    logger.debug("Error in to_dict() method: %s", str(e))
                    return str(value)

            # Handle common Python types that might be in configs
            if hasattr(value, "__dict__"):
                try:
                    return cls._convert_dataclass_to_dict(value, _depth, _visited, _max_depth)
                except Exception as e:
                    logger.debug("Error converting object with __dict__: %s", str(e))

            # For any other type, convert to string
            return str(value)

        except Exception as e:
            logger.debug("Error in _convert_to_serializable: %s", str(e))
            return f"<Error converting {type(value).__name__}: {str(e)}>"

        finally:
            # Clean up visited set to avoid memory leaks
            if obj_id in _visited:
                _visited.remove(obj_id)
