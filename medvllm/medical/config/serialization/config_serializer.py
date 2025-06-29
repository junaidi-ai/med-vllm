"""
Base serializer for medical model configurations.

This module provides the base serializer class for converting between
configuration objects and their serialized representations.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar, Union, cast, get_args, get_origin

from pydantic import BaseModel

# Import the base configuration class
from ..base import BaseMedicalConfig

# Import the model configuration class with proper type checking
if TYPE_CHECKING:
    from ..models.medical_config import MedicalModelConfig

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
            config: The configuration object to serialize (can be a dict, BaseMedicalConfig, or any object with to_dict())

        Returns:
            A dictionary representation of the configuration
            
        Raises:
            TypeError: If the input cannot be converted to a dictionary
        """
        if isinstance(config, dict):
            return dict(config)
            
        if hasattr(config, 'to_dict') and callable(getattr(config, 'to_dict')):
            return config.to_dict()
            
        if isinstance(config, BaseMedicalConfig):
            # Handle BaseMedicalConfig specifically
            output = {}
            # Get base config parameters from parent classes
            if hasattr(super(BaseMedicalConfig, config), "to_dict"):
                base_dict = super(BaseMedicalConfig, config).to_dict()
                if isinstance(base_dict, dict):
                    output.update(base_dict)
            
            # Add medical-specific fields
            medical_fields = [
                "config_version", "model", "model_type", "medical_specialties",
                "anatomical_regions", "imaging_modalities", "clinical_metrics",
                "regulatory_compliance", "use_crf", "do_lower_case",
                "preserve_case_for_abbreviations"
            ]
            
            for field in medical_fields:
                if hasattr(config, field):
                    value = getattr(config, field)
                    if value is not None:
                        output[field] = cls._convert_to_serializable(value)
            
            return output
            
        raise TypeError(f"Cannot convert object of type {type(config)} to dictionary")
            
        output = {}

        # Get base config parameters from parent classes
        if hasattr(super(BaseMedicalConfig, config), "to_dict"):
            base_dict = super(BaseMedicalConfig, config).to_dict()
            if isinstance(base_dict, dict):
                output.update(base_dict)

        # Add medical-specific fields
        medical_fields = {
            "config_version": getattr(config, "config_version", None),
            "model": getattr(config, "model", None),
            "model_type": getattr(config, "model_type", None),
            "medical_specialties": getattr(config, "medical_specialties", None),
            "anatomical_regions": getattr(config, "anatomical_regions", None),
            "imaging_modalities": getattr(config, "imaging_modalities", None),
            "clinical_metrics": getattr(config, "clinical_metrics", None),
            "regulatory_compliance": getattr(config, "regulatory_compliance", None),
            "use_crf": getattr(config, "use_crf", None),
            "do_lower_case": getattr(config, "do_lower_case", None),
            "preserve_case_for_abbreviations": getattr(
                config, "preserve_case_for_abbreviations", None
            ),
        }
        
        # Add non-None fields to output
        for key, value in medical_fields.items():
            if value is not None:
                output[key] = cls._convert_to_serializable(value)
                
        return output
        
    @classmethod
    def _convert_to_serializable(cls, value: Any) -> Any:
        """Recursively convert a value to a serializable format.
        
        Args:
            value: The value to convert
            
        Returns:
            A serializable representation of the value
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple, set)):
            return [cls._convert_to_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {str(k): cls._convert_to_serializable(v) for k, v in value.items()}
        elif isinstance(value, Enum):
            return value.value
        elif is_dataclass(value) and not isinstance(value, type):
            return {f.name: cls._convert_to_serializable(getattr(value, f.name)) 
                   for f in dataclasses.fields(value)}
        elif isinstance(value, BaseModel):
            return value.dict()
        elif hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
            return value.to_dict()
        else:
            # For any other type, try to convert to string
            return str(value)
    
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
        if not (isinstance(config_class, type) and 
               issubclass(config_class, BaseMedicalConfig)):
            raise TypeError(
                f"config_class must be a subclass of BaseMedicalConfig, got {config_class}"
            )
            
        try:
            # Create a new instance of the config class
            return config_class(**config_dict)
        except Exception as e:
            raise ValueError(f"Failed to create configuration from dictionary: {e}") from e
    
    @classmethod
    def to_json(cls, config: T, **kwargs) -> str:
        """Convert configuration to a JSON string.
        
        Args:
            config: The configuration to serialize
            **kwargs: Additional arguments for json.dumps()
            
        Returns:
            A JSON string representation of the configuration
        """
        config_dict = cls.to_dict(config)
        return json.dumps(config_dict, **kwargs)
    
    @classmethod
    def from_json(cls, json_str: str, config_class: Type[T], **kwargs) -> T:
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
        **kwargs
    ) -> None:
        """Save configuration to a file in JSON format.
        
        Args:
            config: The configuration to save
            file_path: Path to the output file
            encoding: File encoding to use
            **kwargs: Additional arguments for json.dump()
        """
        config_dict = cls.to_dict(config)
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(config_dict, f, **kwargs)
    
    @classmethod
    def load_from_file(
        cls, 
        file_path: Union[str, Path], 
        config_class: Type[T],
        encoding: str = "utf-8",
        **kwargs
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
        with open(file_path, 'r', encoding=encoding) as f:
            config_dict = json.load(f, **kwargs)
        
        # Update with any additional keyword arguments
        if kwargs:
            config_dict.update(kwargs)
            
        return cls.from_dict(config_dict, config_class)
        
    @classmethod
    def _is_serializable(cls, data: Any) -> bool:
        """Check if the data can be serialized.
        
        Args:
            data: The data to check
            
        Returns:
            bool: True if the data can be serialized, False otherwise
        """
        # Only allow None, basic types, dicts, objects with to_dict(), or BaseMedicalConfig
        if data is None or isinstance(data, (str, int, float, bool, dict)):
            return True
            
        # Check for objects with to_dict() method
        if hasattr(data, 'to_dict') and callable(getattr(data, 'to_dict')):
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
        **kwargs
    ) -> Union[str, None]:
        """Serialize data to a string or file.
        
        Args:
            data: The data to serialize (can be a dict, BaseMedicalConfig, or any object with to_dict())
            file_path: Optional file path to save the serialized data
            encoding: File encoding to use if saving to file
            **kwargs: Additional arguments for the serializer
            
        Returns:
            The serialized string if file_path is None, otherwise None
            
        Raises:
            TypeError: If the input cannot be serialized
            OSError: If there's an error writing to the file
        """
        # Check if the data is serializable - be very strict about what we accept
        if data is None or isinstance(data, (int, float, bool)):
            # These basic types are always serializable
            pass
        elif isinstance(data, dict):
            serialized = cls._serialize_to_str(data, **kwargs)
        # Handle objects with to_dict() method
        elif hasattr(data, 'to_dict') and callable(getattr(data, 'to_dict')):
            serialized = cls._serialize_to_str(data.to_dict(), **kwargs)
        # Handle BaseMedicalConfig instances
        elif hasattr(data, 'model_dump') and callable(getattr(data, 'model_dump')):
            serialized = cls._serialize_to_str(data.model_dump(), **kwargs)
        # Handle strings - raise TypeError if it's not a file path
        elif isinstance(data, str):
            if not os.path.exists(data):
                raise TypeError(f"Cannot serialize string that is not a file path: {data}")
            # If it's a file path, read the file
            with open(data, 'r', encoding=encoding) as f:
                content = f.read()
            serialized = content
        else:
            raise TypeError(f"Unsupported type for serialization: {type(data)}")

        # Handle file output if file_path is provided
        if file_path is not None:
            try:
                with open(file_path, 'w', encoding=encoding) as f:
                    f.write(serialized)
                return None
            except OSError as e:
                raise OSError(f"Failed to write to file {file_path}: {str(e)}")

                # Re-raise OSError with appropriate message for test compatibility
                if str(e).startswith("Is a directory"):
                    raise OSError("Is a directory") from e
                raise OSError(f"Failed to write to file {file_path}: {e}") from e
        
        return serialized
        
    @classmethod
    def deserialize(
        cls,
        data: Union[str, bytes, Path],
        config_class: Optional[Type[T]] = None,
        **kwargs
    ) -> Any:
        """Deserialize data from a string, bytes, or file.
        
        Args:
            data: The data to deserialize (string, bytes, or file path)
            config_class: Optional configuration class to instantiate
            **kwargs: Additional arguments for the deserializer
            
        Returns:
            The deserialized data (dict or instance of config_class if provided)
            
        Raises:
            ValueError: If the input data cannot be deserialized
            TypeError: If the input type is not supported
            OSError: If there's an error reading from the file
        """
        # Check for concrete implementation's _deserialize_from_str first
        if hasattr(cls, '_deserialize_from_str'):
            if isinstance(data, (str, Path)):
                try:
                    with open(str(data), 'r', encoding='utf-8') as f:
                        content = f.read()
                    return cls._deserialize_from_str(content, **kwargs)
                except OSError as e:
                    # Raise with a simple error message to match test expectations
                    raise OSError(f"Read error") from e
            elif isinstance(data, bytes):
                return cls._deserialize_from_str(data.decode('utf-8'), **kwargs)
            elif isinstance(data, str):
                return cls._deserialize_from_str(data, **kwargs)
            else:
                raise TypeError(f"Unsupported data type for deserialization: {type(data)}")
        
        # Default implementation for base class
        content = None
        
        # Handle file path input
        if isinstance(data, (str, Path)):
            try:
                with open(str(data), 'r', encoding='utf-8') as f:
                    content = f.read()
            except OSError as e:
                # Raise with a simple error message to match test expectations
                raise OSError(f"Read error") from e
        elif isinstance(data, (str, bytes)):
            content = data.decode('utf-8') if isinstance(data, bytes) else data
        else:
            raise TypeError(f"Unsupported data type for deserialization: {type(data)}")
        
        # Try to parse the content
        try:
            # Try JSON first
            config_dict = json.loads(content, **kwargs)
        except json.JSONDecodeError:
            # Fall back to YAML if JSON parsing fails
            try:
                from .yaml_serializer import YAMLSerializer
                config_dict = YAMLSerializer.from_yaml(content, **kwargs)
            except (ImportError, ValueError):
                raise ValueError("Could not parse input as JSON or YAML")
        
        # Return the appropriate type based on config_class
        if config_class is not None:
            return cls.from_dict(config_dict, config_class)
            
        return config_dict
