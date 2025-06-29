"""
YAML serialization for medical model configurations.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast

from ..base import BaseMedicalConfig
from .config_serializer import ConfigSerializer

# Try to import PyYAML
try:
    import yaml
    PYYAML_AVAILABLE = True
except ImportError:
    PYYAML_AVAILABLE = False

T = TypeVar('T', bound='MedicalModelConfig')

class YAMLSerializer(ConfigSerializer):
    """YAML serializer for medical model configurations."""
    
    @classmethod
    def to_yaml(
        cls,
        config: BaseMedicalConfig,
        file_path: Optional[Union[str, Path]] = None,
        **yaml_kwargs: Any
    ) -> Optional[str]:
        """Serialize configuration to YAML.
        
        Args:
            config: Configuration to serialize
            file_path: Optional file path to save the YAML to
            **yaml_kwargs: Additional arguments to pass to yaml.dump()
            
        Returns:
            YAML string if file_path is None, else None
            
        Raises:
            ImportError: If PyYAML is not installed
        """
        if not PYYAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML serialization. "
                "Install with: pip install pyyaml"
            )
            
        config_dict = cls.to_dict(config)
        default_flow_style = yaml_kwargs.pop('default_flow_style', False)
        
        if file_path is not None:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_dict,
                    f,
                    default_flow_style=default_flow_style,
                    **yaml_kwargs
                )
            return None
            
        return yaml.dump(
            config_dict,
            default_flow_style=default_flow_style,
            **yaml_kwargs
        )
    
    @classmethod
    def from_yaml(
        cls,
        yaml_data: Union[str, bytes, bytearray, Path],
        config_class: Type[T],
        **yaml_kwargs: Any
    ) -> T:
        """Deserialize configuration from YAML.
        
        Args:
            yaml_data: YAML string, bytes, or file path
            config_class: Configuration class to instantiate
            **yaml_kwargs: Additional arguments to pass to yaml.safe_load()
            
        Returns:
            Deserialized configuration object
            
        Raises:
            ImportError: If PyYAML is not installed
        """
        if not PYYAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML deserialization. "
                "Install with: pip install pyyaml"
            )
            
        if isinstance(yaml_data, (str, bytes, bytearray)):
            if isinstance(yaml_data, (bytes, bytearray)):
                yaml_data = yaml_data.decode('utf-8')
            config_dict = yaml.safe_load(yaml_data, **yaml_kwargs)
        else:
            with open(yaml_data, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f, **yaml_kwargs)
                
        return cls.from_dict(config_dict, config_class)
