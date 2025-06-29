"""
Serialization and deserialization utilities for medical model configurations.

This module provides a unified interface for serializing and deserializing
medical model configurations to/from various formats like JSON and YAML.
"""

import warnings
from typing import Any, Optional, Type, TypeVar, Union

from ..base import BaseMedicalConfig
from .config_serializer import ConfigSerializer

# Type variable for generic configuration types
T = TypeVar('T', bound=BaseMedicalConfig)

# Import JSON serializer (always available as it uses standard library)
from .json_serializer import JSONSerializer

# Import YAML serializer if PyYAML is available
try:
    from .yaml_serializer import YAMLSerializer, PYYAML_AVAILABLE
    
    if not PYYAML_AVAILABLE:
        warnings.warn(
            "PyYAML is not installed. YAML serialization will not be available. "
            "Install with: pip install pyyaml",
            ImportWarning,
            stacklevel=2
        )
        YAMLSerializer = None  # type: ignore
        
except ImportError:
    YAMLSerializer = None  # type: ignore
    PYYAML_AVAILABLE = False
    warnings.warn(
        "PyYAML is not installed. YAML serialization will not be available. "
        "Install with: pip install pyyaml",
        ImportWarning,
        stacklevel=2
    )


def save_config(
    config: BaseMedicalConfig,
    file_path: str,
    format: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Save configuration to a file.
    
    The format is determined by the file extension if not specified.
    
    Args:
        config: Configuration to save
        file_path: Path to save the configuration to
        format: Optional format ('json' or 'yaml'). If None, inferred from file extension.
        **kwargs: Additional arguments to pass to the serializer
        
    Raises:
        ValueError: If the format is not supported or cannot be inferred
        ImportError: If the required dependencies are not installed
    """
    if format is None:
        if file_path.lower().endswith('.json'):
            format = 'json'
        elif file_path.lower().endswith(('.yaml', '.yml')):
            format = 'yaml'
        else:
            raise ValueError(
                "Could not infer format from file extension. "
                "Please specify format='json' or format='yaml'"
            )
    
    format = format.lower()
    if format == 'json':
        JSONSerializer.save_to_file(config, file_path, **kwargs)
    elif format == 'yaml':
        if YAMLSerializer is None:
            raise ImportError(
                "PyYAML is required for YAML serialization. "
                "Install with: pip install pyyaml"
            )
        YAMLSerializer.to_yaml(config, file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_config(
    file_path: str,
    config_class: Type[T],
    format: Optional[str] = None,
    **kwargs: Any
) -> T:
    """Load configuration from a file.
    
    The format is determined by the file extension if not specified.
    
    Args:
        file_path: Path to the configuration file
        config_class: Configuration class to instantiate
        format: Optional format ('json' or 'yaml'). If None, inferred from file extension.
        **kwargs: Additional arguments to pass to the deserializer
        
    Returns:
        An instance of the specified configuration class
        
    Raises:
        ValueError: If the format is not supported or cannot be inferred
        ImportError: If the required dependencies are not installed
    """
    if format is None:
        if file_path.lower().endswith('.json'):
            format = 'json'
        elif file_path.lower().endswith(('.yaml', '.yml')):
            format = 'yaml'
        else:
            raise ValueError(
                "Could not infer format from file extension. "
                "Please specify format='json' or format='yaml'"
            )
    
    format = format.lower()
    if format == 'json':
        return JSONSerializer.load_from_file(file_path, config_class, **kwargs)
    elif format == 'yaml':
        if YAMLSerializer is None:
            raise ImportError(
                "PyYAML is required for YAML deserialization. "
                "Install with: pip install pyyaml"
            )
        return YAMLSerializer.from_yaml(file_path, config_class, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


__all__ = [
    "ConfigSerializer",
    "JSONSerializer",
    "YAMLSerializer",
    "save_config",
    "load_config",
]
