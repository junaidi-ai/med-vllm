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
from typing import Any, Dict, Optional, Type, TypeVar, Union, overload, TYPE_CHECKING

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
    ) -> str:
        ...
        
    @classmethod
    @overload
    def to_json(
        cls,
        config: T,
        file_path: Union[str, Path],
        **json_kwargs: Any,
    ) -> None:
        ...

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
        # Get default kwargs if not provided
        json_kwargs.setdefault("indent", 2)
        json_kwargs.setdefault("ensure_ascii", False)
        
        # Convert config to dictionary
        config_dict = cls.to_dict(config)

        # Write to file if file_path is provided
        if file_path is not None:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, **json_kwargs)
                return None
            except (IOError, OSError) as e:
                raise ValueError(f"Failed to write to file {file_path}: {e}") from e
            except (TypeError, ValueError) as e:
                raise ValueError(f"Failed to serialize configuration: {e}") from e

        # Return as string if no file path provided
        try:
            return json.dumps(config_dict, **json_kwargs)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize configuration: {e}") from e

    @classmethod
    @overload
    def from_json(
        cls,
        json_data: Union[str, bytes, bytearray],
        config_class: Type[T],
        **json_kwargs: Any,
    ) -> T:
        ...
        
    @classmethod
    @overload
    def from_json(
        cls,
        json_data: Path,
        config_class: Type[T],
        **json_kwargs: Any,
    ) -> T:
        ...

    @classmethod
    def from_json(
        cls,
        json_data: Union[str, bytes, bytearray, Path],
        config_class: Type[T],
        **json_kwargs: Any,
    ) -> T:
        """Deserialize configuration from JSON.

        Args:
            json_data: JSON string, bytes, or path to a JSON file
            config_class: Configuration class to instantiate
            **json_kwargs: Additional arguments to pass to json.loads() or json.load()

        Returns:
            An instance of the specified configuration class

        Raises:
            TypeError: If json_data is not a valid type
            ValueError: If deserialization fails or the data is invalid
            FileNotFoundError: If json_data is a path that doesn't exist
            json.JSONDecodeError: If the JSON data is malformed
        """
        # Set default kwargs if not provided
        json_kwargs.setdefault("parse_float", float)
        json_kwargs.setdefault("parse_int", int)
        json_kwargs.setdefault("parse_constant", str)
        
        try:
            # Handle file path input
            if isinstance(json_data, (str, Path)) and str(json_data).endswith(".json"):
                try:
                    with open(json_data, "r", encoding="utf-8") as f:
                        config_dict = json.load(f, **json_kwargs)
                except FileNotFoundError as e:
                    # If it's a string that doesn't exist as a file, try parsing as JSON
                    if isinstance(json_data, str):
                        try:
                            config_dict = json.loads(json_data, **json_kwargs)
                        except json.JSONDecodeError:
                            # Re-raise the original FileNotFoundError if parsing fails
                            raise FileNotFoundError(
                                f"File not found and input is not valid JSON: {e}"
                            ) from e
                    else:
                        # Re-raise if it's a Path that doesn't exist
                        raise
            # Handle string/bytes input
            elif isinstance(json_data, (str, bytes, bytearray)):
                if isinstance(json_data, bytes):
                    json_data = json_data.decode("utf-8")
                config_dict = json.loads(json_data, **json_kwargs)
            else:
                raise TypeError(
                    f"Expected str, bytes, or Path, got {type(json_data).__name__}"
                )
            
            # Create configuration object from dictionary
            return cls.from_dict(config_dict, config_class)
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse JSON data: {e.msg}", 
                e.doc, 
                e.pos
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to deserialize configuration: {e}") from e
            
    @classmethod
    def _convert_to_serializable(cls, value: Any) -> Any:
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
            A JSON-serializable representation of the value
            
        Raises:
            ValueError: If the value cannot be converted to a JSON-serializable format
        """
        # Handle None and basic types
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
            
        # Handle collections
        if isinstance(value, (list, tuple, set)):
            return [cls._convert_to_serializable(item) for item in value]
            
        if isinstance(value, dict):
            return {str(k): cls._convert_to_serializable(v) for k, v in value.items()}
            
        # Handle enums
        if isinstance(value, Enum):
            return value.value
            
        # Handle dataclasses
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                f.name: cls._convert_to_serializable(getattr(value, f.name)) 
                for f in dataclasses.fields(value)
            }
            
        # Handle Pydantic models
        if isinstance(value, BaseModel):
            return value.dict()
            
        # Handle objects with to_dict method
        if hasattr(value, 'to_dict') and callable(value.to_dict):
            result = value.to_dict()
            # Ensure the result is serializable
            return cls._convert_to_serializable(result)
            
        # For any other type, try to convert to string
        try:
            return str(value)
        except Exception as e:
            raise ValueError(
                f"Could not convert value of type {type(value).__name__} to a JSON-serializable format: {e}"
            ) from e
