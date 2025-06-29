"""
JSON serialization for medical model configurations.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast

from ..base import BaseMedicalConfig
from .config_serializer import ConfigSerializer

T = TypeVar('T', bound='MedicalModelConfig')

class JSONSerializer(ConfigSerializer):
    """JSON serializer for medical model configurations."""
    
    @classmethod
    def to_json(
        cls,
        config: BaseMedicalConfig,
        file_path: Optional[Union[str, Path]] = None,
        **json_kwargs: Any
    ) -> Optional[str]:
        """Serialize configuration to JSON.
        
        Args:
            config: Configuration to serialize
            file_path: Optional file path to save the JSON to
            **json_kwargs: Additional arguments to pass to json.dump()
            
        Returns:
            JSON string if file_path is None, else None
        """
        config_dict = cls.to_dict(config)
        
        if file_path is not None:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, **json_kwargs)
            return None
            
        return json.dumps(config_dict, indent=2, **json_kwargs)
    
    @classmethod
    def from_json(
        cls,
        json_data: Union[str, bytes, bytearray, Path],
        config_class: Type[T],
        **json_kwargs: Any
    ) -> T:
        """Deserialize configuration from JSON.
        
        Args:
            json_data: JSON string, bytes, or file path
            config_class: Configuration class to instantiate
            **json_kwargs: Additional arguments to pass to json.load()
            
        Returns:
            Deserialized configuration object
        """
        if isinstance(json_data, (str, bytes, bytearray)):
            if isinstance(json_data, (bytes, bytearray)):
                json_data = json_data.decode('utf-8')
            config_dict = json.loads(json_data, **json_kwargs)
        else:
            with open(json_data, 'r', encoding='utf-8') as f:
                config_dict = json.load(f, **json_kwargs)
                
        return cls.from_dict(config_dict, config_class)
