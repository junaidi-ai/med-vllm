"""
YAML serialization for medical model configurations.

This module provides YAML serialization and deserialization for medical
model configurations, with support for both file and string I/O.

The module requires PyYAML to be installed for YAML support. If PyYAML is not
available, the YAMLSerializer class will not be usable.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast, overload

from ..base import BaseMedicalConfig
from .config_serializer import ConfigSerializer

# Try to import PyYAML
try:
    import yaml
    from yaml import SafeDumper, SafeLoader, YAMLError
    PYYAML_AVAILABLE = True
except ImportError:
    PYYAML_AVAILABLE = False
    # Create a dummy YAMLError for type checking when PyYAML is not available
    class YAMLError(Exception):  # type: ignore
        """Dummy YAMLError class when PyYAML is not available."""
        pass

# Type variable for generic configuration types
T = TypeVar("T", bound=BaseMedicalConfig)

# Re-export YAMLError for consistent error handling
__all__ = ["YAMLSerializer", "PYYAML_AVAILABLE", "YAMLError"]


class YAMLSerializer(ConfigSerializer):
    """YAML serializer for medical model configurations.
    
    This class provides methods for serializing and deserializing
    configuration objects to/from YAML format, with support for both
    file and string I/O.
    """
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if YAML serialization is available (PyYAML is installed).
        
        Returns:
            bool: True if PyYAML is available, False otherwise.
        """
        return PYYAML_AVAILABLE

    @classmethod
    def to_yaml(
        cls,
        config: Union[T, Dict[str, Any]],
        file_path: Optional[Union[str, Path]] = None,
        **yaml_kwargs: Any,
    ) -> Optional[str]:
        """Serialize configuration to YAML.

        Args:
            config: Configuration to serialize (dict or BaseMedicalConfig)
            file_path: Optional file path to save the YAML to. If None,
                     returns the YAML as a string.
            **yaml_kwargs: Additional arguments to pass to yaml.dump()

        Returns:
            YAML string if file_path is None, else None
            
        Raises:
            ImportError: If PyYAML is not installed
            ValueError: If serialization fails
        """
        if not PYYAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML serialization. "
                "Install with: pip install pyyaml"
            )

        try:
            # Convert config to dict if it's a BaseMedicalConfig instance
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            else:
                config_dict = dict(config)
            # Set default YAML options if not provided
            yaml_kwargs.setdefault('sort_keys', False)
            yaml_kwargs.setdefault('allow_unicode', True)
            
            # Dump to YAML string with explicit Dumper to control flow style
            class CustomDumper(yaml.SafeDumper):
                pass
                
            def dict_representer(dumper, data):
                return dumper.represent_dict(data.items())
                
            CustomDumper.add_representer(dict, dict_representer)
            
            # Use flow_style parameter to control formatting
            yaml_str = yaml.dump(
                config_dict, 
                Dumper=CustomDumper,
                default_flow_style=yaml_kwargs.pop('default_flow_style', None),
                **yaml_kwargs
            )
            
            # Write to file if path is provided
            if file_path is not None:
                file_path = Path(file_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(yaml_str)
                return None
                
            return yaml_str
            
        except Exception as e:
            raise ValueError(f"Failed to serialize configuration: {e}") from e

    @classmethod
    def from_yaml(
        cls,
        yaml_data: Union[str, bytes, Path],
        config_class: Optional[Type[T]] = None,
        **yaml_kwargs: Any,
    ) -> Union[Dict[str, Any], T]:
        """Deserialize configuration from YAML.

        Args:
            yaml_data: YAML string, bytes, or path to YAML file
            config_class: Optional configuration class to deserialize into.
                        If None, returns a dictionary.
            **yaml_kwargs: Additional arguments to pass to yaml.safe_load()

        Returns:
            Deserialized configuration object or dictionary
            
        Raises:
            ImportError: If PyYAML is not installed
            ValueError: If deserialization fails or the data is invalid
        """
        if not PYYAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML deserialization. "
                "Install with: pip install pyyaml"
            )

        config_dict: Optional[Dict[str, Any]] = None
        
        try:
            # Handle Path objects
            if isinstance(yaml_data, Path):
                if not yaml_data.exists():
                    raise ValueError(f"File not found: {yaml_data}")
                with open(yaml_data, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f, **yaml_kwargs)
            # Handle file-like paths that exist
            elif (isinstance(yaml_data, str) and 
                  len(yaml_data) < 260 and  # Reasonable path length
                  os.path.isfile(yaml_data)):
                with open(yaml_data, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f, **yaml_kwargs)
            # Handle string/bytes input
            else:
                try:
                    # Convert to string if it's bytes
                    yaml_str = yaml_data.decode('utf-8') if isinstance(yaml_data, bytes) else yaml_data
                    # Try to parse as YAML
                    config_dict = yaml.safe_load(yaml_str, **yaml_kwargs)
                except yaml.YAMLError as e:
                    # If it looks like a file path but doesn't exist, provide a better error
                    if (isinstance(yaml_data, str) and 
                        len(yaml_data) < 260 and 
                        any(c in yaml_data for c in '/\\')):
                        raise ValueError(f"File not found or invalid YAML: {yaml_data}")
                    raise ValueError(f"Invalid YAML: {e}")
            
            # If we got here but config_dict is None, the YAML was empty or None
            if config_dict is None:
                raise ValueError("Empty YAML content")
                
            if config_class is None:
                return config_dict
                
            # Create config instance if config_class is provided
            return config_class(**config_dict)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")
        except Exception as e:
            raise ValueError(f"Failed to deserialize configuration: {e}")
