"""
Serialization and deserialization utilities for medical model configurations.

This module provides a unified interface for serializing and deserializing
medical model configurations to/from various formats like JSON and YAML.
"""

import warnings
from typing import Any, Optional, Type, TypeVar

from ..base import BaseMedicalConfig
from .config_serializer import ConfigSerializer
from .json_serializer import JSONSerializer

# Type variable for generic configuration types
T = TypeVar("T", bound=BaseMedicalConfig)

# Import YAML serializer if PyYAML is available
# Initialize YAML-related variables
try:
    from .yaml_serializer import PYYAML_AVAILABLE
    from .yaml_serializer import YAMLSerializer as YAML

    YAML_AVAILABLE = PYYAML_AVAILABLE
    YAMLSerializer = YAML if YAML_AVAILABLE else None
except ImportError:
    YAML_AVAILABLE = False
    YAMLSerializer = None

yaml_warning = (
    "PyYAML is not installed. YAML serialization will not be available. "
    "Install with: pip install pyyaml"
)

if not YAML_AVAILABLE:
    warnings.warn(yaml_warning, ImportWarning, stacklevel=2)


def save_config(
    config: BaseMedicalConfig,
    file_path: str,
    format: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Save configuration to a file.

    The format is determined by the file extension if not specified.

    Args:
        config: Configuration to save
        file_path: Path to save the configuration to
        format: Optional format ('json' or 'yaml'). If None, inferred
               from file extension.
        **kwargs: Additional arguments to pass to the serializer

    Raises:
        ValueError: If the format is not supported or cannot be inferred
        ImportError: If the required dependencies are not installed
    """
    if format is None:
        if file_path.lower().endswith(".json"):
            format = "json"
        elif file_path.lower().endswith((".yaml", ".yml")):
            format = "yaml"
        else:
            raise ValueError(
                "Could not infer format from file extension. "
                "Please specify format='json' or format='yaml'"
            )

    format = format.lower()
    if format == "json":
        JSONSerializer.save_to_file(config, file_path, **kwargs)
    elif format == "yaml":
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
    **kwargs: Any,
) -> T:
    """Load configuration from a file.

    The format is determined by the file extension if not specified.

    Args:
        file_path: Path to the configuration file
        config_class: Configuration class to instantiate
        format: Optional format ('json' or 'yaml'). If None, inferred
               from file extension.
        **kwargs: Additional arguments to pass to the deserializer

    Returns:
        An instance of the specified configuration class

    Raises:
        ValueError: If the format is not supported or cannot be inferred,
                   or if the deserialized data is invalid
        ImportError: If the required dependencies are not installed
    """
    # Determine format from file extension if not specified
    if format is None:
        if file_path.lower().endswith(".json"):
            file_format = "json"
        elif file_path.lower().endswith((".yaml", ".yml")):
            file_format = "yaml"
        else:
            raise ValueError(
                "Could not infer format from file extension. "
                "Please specify format='json' or format='yaml'"
            )
    else:
        file_format = format.lower()

    # Validate format
    if file_format not in ("json", "yaml"):
        raise ValueError(f"Unsupported format: {file_format}")

    # Load and process the configuration based on format
    if file_format == "json":
        return _load_json_config(file_path, config_class, **kwargs)
    else:
        return _load_yaml_config(file_path, config_class, **kwargs)


def _load_json_config(file_path: str, config_class: Type[T], **kwargs: Any) -> T:
    """Load configuration from a JSON file.

    Args:
        file_path: Path to the JSON configuration file
        config_class: Configuration class to instantiate
        **kwargs: Additional arguments to pass to the deserializer

    Returns:
        An instance of the specified configuration class

    Raises:
        ValueError: If the deserialized data is invalid
    """
    result = JSONSerializer.load_from_file(file_path, config_class, **kwargs)
    if isinstance(result, dict):
        return config_class(**result)
    if isinstance(result, config_class):
        return result
    raise ValueError(
        "Unexpected result type from JSON deserialization: "
        f"{type(result).__name__}. Expected dict or {config_class.__name__}."
    )


def _load_yaml_config(file_path: str, config_class: Type[T], **kwargs: Any) -> T:
    """Load configuration from a YAML file.

    Args:
        file_path: Path to the YAML configuration file
        config_class: Configuration class to instantiate
        **kwargs: Additional arguments to pass to the deserializer

    Returns:
        An instance of the specified configuration class

    Raises:
        ImportError: If PyYAML is not installed
        ValueError: If the deserialized data is invalid
    """
    if YAMLSerializer is None:
        raise ImportError(
            "PyYAML is required for YAML deserialization. "
            "Install with: pip install pyyaml"
        )

    result = YAMLSerializer.from_yaml(file_path, config_class, **kwargs)
    if isinstance(result, dict):
        return config_class(**result)
    if isinstance(result, config_class):
        return result
    raise ValueError(
        "Unexpected result type from YAML deserialization: "
        f"{type(result).__name__}. Expected dict or {config_class.__name__}."
    )


__all__ = [
    "ConfigSerializer",
    "JSONSerializer",
    "YAMLSerializer",
    "save_config",
    "load_config",
]
