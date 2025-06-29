"""
Base configuration classes for medical models.

This module provides the BaseMedicalConfig class which serves as the foundation
for all medical model configurations in the medvllm library.
"""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from medvllm.config import Config

CONFIG_VERSION = "1.0.0"


@dataclass
class BaseMedicalConfig(Config):
    """Base configuration class for medical models.

    This class provides common functionality and fields for medical model configurations.
    It inherits from the base Config class and adds medical-specific features.

    Attributes:
        config_version: Version of the configuration schema
        file_path: Path to the config file if loaded from disk
        model_type: Type of the model (default: "base")
    """
    config_version: str = field(default=CONFIG_VERSION, init=False)
    file_path: Optional[str] = field(default=None)  # Path to the config file if loaded from disk
    model_type: str = field(default="base")  # Default model type

    def __init__(self, **kwargs):
        """Initialize the configuration with support for extra fields."""
        # Call parent's __init__ with known parameters
        known_params = {k: v for k, v in kwargs.items() 
                      if k in {'model', 'max_num_batched_tokens', 'max_num_seqs', 
                              'max_model_len', 'gpu_memory_utilization', 
                              'tensor_parallel_size', 'enforce_eager', 
                              'hf_config', 'eos', 'kvcache_block_size', 
                              'num_kvcache_blocks'}}
        super().__init__(**known_params)
        
        # Set other attributes
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
                
        # Ensure model_type is set
        if not hasattr(self, 'model_type') or self.model_type is None:
            self.model_type = "base"

    def __setattr__(self, name, value):
        """Allow setting arbitrary attributes."""
        object.__setattr__(self, name, value)

    def __post_init__(self):
        """Initialize the configuration with default values and validate."""
        # Call parent's __post_init__ first
        try:
            super().__post_init__()
        except AttributeError as e:
            # If parent's __post_init__ fails, we'll handle it in validate()
            pass
            
        # Set default values
        if not hasattr(self, 'model_type') or not self.model_type:
            self.model_type = 'base'
            
        self._extra_fields = {}
        
        # Validate the configuration
        self.validate()

    def ensure_compatibility(self) -> bool:
        """Ensure the configuration is compatible with the current version."""
        if not hasattr(self, "config_version") or self.config_version != CONFIG_VERSION:
            self._migrate_config()
            return False
        return True

    def _migrate_config(self) -> None:
        """Migrate the configuration to the latest version."""
        version = getattr(self, "config_version", "0.1.0")
        if version == "0.1.0":
            if hasattr(self, "medical_params") and isinstance(self.medical_params, dict):
                for key, value in self.medical_params.items():
                    if not hasattr(self, key):
                        setattr(self, key, value)
                delattr(self, "medical_params")
            self.config_version = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        # Get all non-private attributes
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMedicalConfig':
        """Create a configuration from a dictionary.
        
        Args:
            data: Dictionary containing configuration parameters
            
        Returns:
            New instance of the configuration
        """
        # Create a copy of the data to avoid modifying the input
        data_copy = data.copy()
        
        # Remove model_type from the data if it exists
        model_type = data_copy.pop('model_type', None)
        
        # Create the instance with the remaining data
        instance = cls(**data_copy)
        
        # Set the model_type if it was provided
        if model_type is not None:
            instance.model_type = model_type
            
        return instance
        
    def update_from_dict(self, data: Dict[str, Any]) -> 'BaseMedicalConfig':
        """Update the configuration from a dictionary.
        
        Args:
            data: Dictionary containing configuration parameters
            
        Returns:
            The updated instance
        """
        for key, value in data.items():
            setattr(self, key, value)
        return self
    
    def copy(self) -> 'BaseMedicalConfig':
        """Create a copy of the configuration.
        
        Returns:
            A new instance with the same parameters
        """
        # Get the dictionary representation
        data = self.to_dict()
        
        # Create a new instance using from_dict to ensure proper initialization
        return self.__class__.from_dict(data)
    
    def validate(self) -> None:
        """Validate the configuration."""
        # Check required fields
        required_fields = ['model_type']
        for field in required_fields:
            if not hasattr(self, field) or getattr(self, field) is None:
                raise ValueError(f"Missing required field: {field}")
        
        # Try to set up hf_config if not already set
        if hasattr(self, 'model') and hasattr(self, 'hf_config') and self.hf_config is None:
            try:
                from transformers import AutoConfig
                self.hf_config = AutoConfig.from_pretrained(self.model)
                if hasattr(self, 'max_model_len') and hasattr(self.hf_config, 'max_position_embeddings'):
                    self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
            except Exception:
                # It's okay if we can't load the config, we'll just skip setting max_model_len
                pass
                
        # Call custom validation if implemented
        if hasattr(self, '_validate_custom'):
            self._validate_custom()
