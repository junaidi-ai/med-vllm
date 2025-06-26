"""
Versioning and migration utilities for medical model configurations.

This module handles version checking and migration of configuration files.
"""

from typing import Any, Dict, Optional, Type, TypeVar

from .base import BaseMedicalConfig

T = TypeVar('T', bound='MedicalModelConfig')

class ConfigVersioner:
    """Handles version checking and migration of configuration files."""
    
    @classmethod
    def get_schema_version(cls) -> str:
        """Get the current schema version."""
        from .base import CONFIG_VERSION
        return CONFIG_VERSION
    
    @classmethod
    def migrate_config(cls, config: BaseMedicalConfig) -> None:
        """Migrate a configuration to the latest version."""
        if not hasattr(config, 'config_version'):
            config.config_version = '0.1.0'
            
        if config.config_version == '0.1.0':
            cls._migrate_0_1_to_1_0(config)
    
    @staticmethod
    def _migrate_0_1_to_1_0(config: BaseMedicalConfig) -> None:
        """Migrate from version 0.1.0 to 1.0.0."""
        if hasattr(config, 'medical_params') and isinstance(config.medical_params, dict):
            for key, value in config.medical_params.items():
                if not hasattr(config, key):
                    setattr(config, key, value)
            delattr(config, 'medical_params')
        config.config_version = '1.0.0'
