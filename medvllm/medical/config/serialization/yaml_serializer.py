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
    @overload
    def to_yaml(
        cls,
        config: T,
        file_path: None = None,
        **yaml_kwargs: Any,
    ) -> str:
        ...
        
    @classmethod
    @overload
    def to_yaml(
        cls,
        config: T,
        file_path: Union[str, Path],
        **yaml_kwargs: Any,
    ) -> None:
        ...

    @classmethod
    def to_yaml(
        cls,
        config: T,
        file_path: Optional[Union[str, Path]] = None,
        **yaml_kwargs: Any,
    ) -> Optional[str]:
        """Serialize configuration to YAML.

        Args:
            config: Configuration to serialize
            file_path: Optional file path to save the YAML to. If None,
                     returns the YAML as a string.
            **yaml_kwargs: Additional arguments to pass to yaml.dump()

        Returns:
            YAML string if file_path is None, else None
            
        Raises:
            ImportError: If PyYAML is not installed
            TypeError: If the input is not a valid configuration object
            ValueError: If serialization fails
        """
        if not PYYAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML serialization. "
                "Install with: pip install pyyaml"
            )
            
        # Set default kwargs if not provided
        yaml_kwargs.setdefault("default_flow_style", False)
        yaml_kwargs.setdefault("sort_keys", False)
        yaml_kwargs.setdefault("allow_unicode", True)
        yaml_kwargs.setdefault("encoding", "utf-8")
        
        # Convert config to dictionary
        config_dict = cls.to_dict(config)
        
        # Configure YAML dumper to handle special types
        def default_representer(dumper: SafeDumper, data: Any) -> Any:
            if hasattr(data, 'to_dict'):
                return dumper.represent_dict(data.to_dict())
            elif hasattr(data, '_asdict'):  # For namedtuples
                return dumper.represent_dict(data._asdict())
            elif hasattr(data, '__dict__'):  # For objects with __dict__
                return dumper.represent_dict(data.__dict__)
            return dumper.represent_data(data)
            
        yaml.add_representer(object, default_representer, Dumper=SafeDumper)

        # Write to file if file_path is provided
        if file_path is not None:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, Dumper=SafeDumper, **yaml_kwargs)
                return None
            except (IOError, OSError) as e:
                raise ValueError(f"Failed to write to file {file_path}: {e}") from e
            except (TypeError, ValueError) as e:
                raise ValueError(f"Failed to serialize configuration: {e}") from e

        # Return as string if no file path provided
        try:
            return yaml.dump(config_dict, Dumper=SafeDumper, **yaml_kwargs)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize configuration: {e}") from e

    @classmethod
    @overload
    def from_yaml(
        cls,
        yaml_data: Union[str, bytes, bytearray],
        config_class: Type[T],
        **yaml_kwargs: Any,
    ) -> T:
        ...
        
    @classmethod
    @overload
    def from_yaml(
        cls,
        yaml_data: Path,
        config_class: Type[T],
        **yaml_kwargs: Any,
    ) -> T:
        ...

    @classmethod
    def from_yaml(
        cls,
        yaml_data: Union[str, bytes, bytearray, Path],
        config_class: Type[T],
        **yaml_kwargs: Any,
    ) -> T:
        """Deserialize configuration from YAML.

        Args:
            yaml_data: YAML string, bytes, or path to a YAML file
            config_class: Configuration class to instantiate
            **yaml_kwargs: Additional arguments to pass to yaml.safe_load()

        Returns:
            An instance of the specified configuration class
            
        Raises:
            ImportError: If PyYAML is not installed
            TypeError: If yaml_data is not a valid type
            ValueError: If deserialization fails or the data is invalid
            FileNotFoundError: If yaml_data is a path that doesn't exist
            yaml.YAMLError: If the YAML data is malformed
        """
        if not PYYAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML serialization. "
                "Install with: pip install pyyaml"
            )
            
        # Set default kwargs if not provided
        yaml_kwargs.setdefault("Loader", SafeLoader)
        
        def construct_yaml_map(loader: SafeLoader, node: yaml.nodes.MappingNode) -> Dict[Any, Any]:
            """Helper function to properly construct YAML mappings."""
            loader.flatten_mapping(node)
            return dict(loader.construct_pairs(node))
            
        # Register custom constructors for safe loading
        yaml.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_yaml_map,
            Loader=SafeLoader
        )
        
        try:
            # Handle file path input
            if isinstance(yaml_data, (str, Path)) and (str(yaml_data).endswith(('.yaml', '.yml')) or '\n' not in str(yaml_data)):
                try:
                    with open(yaml_data, 'r', encoding='utf-8') as f:
                        config_dict = yaml.safe_load(f, **yaml_kwargs)
                except FileNotFoundError as e:
                    # If it's a string that doesn't exist as a file, try parsing as YAML
                    if isinstance(yaml_data, str):
                        try:
                            config_dict = yaml.safe_load(yaml_data, **yaml_kwargs)
                        except yaml.YAMLError:
                            # Re-raise the original FileNotFoundError if parsing fails
                            raise FileNotFoundError(
                                f"File not found and input is not valid YAML: {e}"
                            ) from e
                    else:
                        # Re-raise if it's a Path that doesn't exist
                        raise
            # Handle string/bytes input
            elif isinstance(yaml_data, (str, bytes, bytearray)):
                if isinstance(yaml_data, (bytes, bytearray)):
                    yaml_data = yaml_data.decode('utf-8')
                config_dict = yaml.safe_load(yaml_data, **yaml_kwargs)
            else:
                raise TypeError(
                    f"Expected str, bytes, or Path, got {type(yaml_data).__name__}"
                )
            
            # Create configuration object from dictionary
            if not isinstance(config_dict, dict):
                raise ValueError(f"Expected a dictionary from YAML, got {type(config_dict).__name__}")
                
            return cls.from_dict(config_dict, config_class)
            
        except yaml.YAMLError as e:
            if hasattr(e, 'problem_mark'):
                mark = e.problem_mark
                raise yaml.YAMLError(
                    f"YAML error at line {mark.line+1}, column {mark.column+1}: {e}"
                ) from e
            raise yaml.YAMLError(f"YAML error: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to deserialize configuration: {e}") from e
