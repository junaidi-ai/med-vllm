"""
Configuration version management.

This module provides version management for configuration objects,
including version checking and migration utilities.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Type, TypeVar, Union, Any
import warnings

T = TypeVar('T')

class ConfigVersionStatus(Enum):
    """Status of a configuration version."""
    CURRENT = "current"
    DEPRECATED = "deprecated"
    UNSUPPORTED = "unsupported"

@dataclass
class ConfigVersionInfo:
    """Information about a configuration version."""
    
    version: str
    status: ConfigVersionStatus
    message: str = ""
    migration_func: Optional[callable] = None

class ConfigVersioner:
    """Manages configuration versions and their status."""
    
    VERSIONS: Dict[str, ConfigVersionInfo] = {
        "1.0.0": ConfigVersionInfo(
            version="1.0.0",
            status=ConfigVersionStatus.CURRENT,
            message="Initial stable release"
        ),
        "0.9.0": ConfigVersionInfo(
            version="0.9.0",
            status=ConfigVersionStatus.DEPRECATED,
            message="Use version 1.0.0 instead",
            migration_func=lambda x: _migrate_090_to_100(x)
        ),
        "0.8.0": ConfigVersionInfo(
            version="0.8.0",
            status=ConfigVersionStatus.UNSUPPORTED,
            message="Version no longer supported, please upgrade to 1.0.0"
        )
    }
    
    CURRENT_VERSION = "1.0.0"
    
    @classmethod
    def get_version_info(cls, version: str) -> ConfigVersionInfo:
        """Get information about a specific version.
        
        Args:
            version: The version string to look up.
            
        Returns:
            ConfigVersionInfo for the specified version.
            
        Raises:
            ValueError: If the version is not found.
        """
        if version not in cls.VERSIONS:
            raise ValueError(f"Unknown version: {version}")
        return cls.VERSIONS[version]
    
    @classmethod
    def check_version_compatibility(cls, version: str) -> None:
        """Check if a version is compatible and issue warnings if deprecated.
        
        Args:
            version: The version string to check.
            
        Raises:
            ValueError: If the version is unsupported.
            UserWarning: If the version is deprecated.
        """
        try:
            version_info = cls.get_version_info(version)
            
            if version_info.status == ConfigVersionStatus.UNSUPPORTED:
                raise ValueError(
                    f"Version {version} is no longer supported. {version_info.message}"
                )
            elif version_info.status == ConfigVersionStatus.DEPRECATED:
                warnings.warn(
                    f"Version {version} is deprecated. {version_info.message}",
                    DeprecationWarning,
                    stacklevel=2
                )
                
        except ValueError as e:
            if version != cls.CURRENT_VERSION:
                raise ValueError(
                    f"Unsupported version: {version}. "
                    f"Current version is {cls.CURRENT_VERSION}"
                ) from e
            raise
    
    @classmethod
    def migrate_config(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate a configuration dictionary to the current version.
        
        Args:
            config_dict: The configuration dictionary to migrate.
            
        Returns:
            The migrated configuration dictionary.
            
        Raises:
            ValueError: If migration is not possible.
        """
        version = config_dict.get('config_version', '0.0.0')
        
        if version == cls.CURRENT_VERSION:
            return config_dict
            
        # Check if version is supported
        if version not in cls.VERSIONS:
            raise ValueError(
                f"Cannot migrate from unsupported version: {version}"
            )
            
        # Get migration path
        versions = sorted(
            [v for v in cls.VERSIONS.keys() if v > version],
            key=lambda v: [int(part) for part in v.split('.')]
        )
        
        current_config = config_dict
        
        # Apply migrations in order
        for target_version in versions:
            version_info = cls.VERSIONS[target_version]
            if version_info.migration_func:
                current_config = version_info.migration_func(current_config)
                current_config['config_version'] = target_version
        
        return current_config

# Migration functions
def _migrate_090_to_100(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from version 0.9.0 to 1.0.0."""
    # Example migration: rename 'model_path' to 'model_name_or_path'
    if 'model_path' in config_dict and 'model_name_or_path' not in config_dict:
        config_dict['model_name_or_path'] = config_dict.pop('model_path')
    
    # Add any other necessary migrations here
    return config_dict
