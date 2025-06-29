"""
Base serializer for medical model configurations.

This module provides the base serializer class for converting between
configuration objects and their serialized representations.
"""

from __future__ import annotations

import json
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
    def to_dict(cls, config: T) -> Dict[str, Any]:
        """Convert a configuration object to a dictionary.

        Args:
            config: The configuration object to serialize

        Returns:
            A dictionary representation of the configuration
            
        Raises:
            TypeError: If the input is not a valid configuration object
        """
        if not isinstance(config, BaseMedicalConfig):
            raise TypeError(f"Expected a BaseMedicalConfig instance, got {type(config)}")
            
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
        return cls.from_dict(config_dict, config_class)

        # Update with any additional keyword arguments
        config_dict.update(kwargs)

        return config_class(**config_dict)
