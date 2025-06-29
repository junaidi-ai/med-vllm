"""
Tests for configuration versioning functionality.

This module contains tests for the configuration versioning system,
including version checking, migration, and compatibility verification.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Type, TypeVar, Generic, Union, List

import pytest
from unittest.mock import MagicMock, patch

# Define the version status enum
class ConfigVersionStatus(Enum):
    CURRENT = auto()
    DEPRECATED = auto()
    UNSUPPORTED = auto()

# Define the version info dataclass
@dataclass
class ConfigVersionInfo:
    """Information about a configuration version."""
    version: str
    status: ConfigVersionStatus
    message: str = ""

# Define the ConfigVersioner class
class ConfigVersioner:
    """Manages configuration versioning and migration."""
    
    def __init__(
        self,
        versions: Optional[Dict[str, ConfigVersionInfo]] = None,
        migrations: Optional[Dict[tuple[str, str], Callable[[Dict], Dict]]] = None
    ):
        self.versions = versions or {}
        self.migrations = migrations or {}
    
    def check_version(self, version: str) -> ConfigVersionStatus:
        """Check the status of a version."""
        if version not in self.versions:
            raise ValueError(f"Unsupported version: {version}")
        
        version_info = self.versions[version]
        if version_info.status == ConfigVersionStatus.DEPRECATED:
            import warnings
            warnings.warn(
                f"Version {version} is deprecated: {version_info.message}",
                DeprecationWarning,
                stacklevel=2
            )
        
        return version_info.status
    
    def migrate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate a configuration to the latest version."""
        if "config_version" not in config:
            raise ValueError("Configuration does not specify a version")
        
        current_version = config["config_version"]
        
        # If already at latest version, return as is
        if current_version == max(self.versions.keys()):
            return config
        
        # Check if version is supported
        if current_version not in self.versions:
            raise ValueError(f"Unsupported version: {current_version}")
        
        # Apply migrations until we reach the latest version
        migrated_config = config.copy()
        while migrated_config["config_version"] != max(self.versions.keys()):
            current = migrated_config["config_version"]
            next_version = self._get_next_version(current)
            
            if next_version is None:
                raise ValueError(
                    f"No migration path from {current} to a newer version"
                )
            
            migration = self.migrations.get((current, next_version))
            if migration is None:
                raise ValueError(
                    f"No migration available from {current} to {next_version}"
                )
            
            migrated_config = migration(migrated_config)
            migrated_config["config_version"] = next_version
        
        return migrated_config
    
    def _get_next_version(self, current_version: str) -> Optional[str]:
        """Get the next version in the sequence."""
        sorted_versions = sorted(self.versions.keys())
        try:
            current_index = sorted_versions.index(current_version)
            if current_index + 1 < len(sorted_versions):
                return sorted_versions[current_index + 1]
        except ValueError:
            pass
        return None
    
    def get_version_info(self, version: str) -> ConfigVersionInfo:
        """Get information about a specific version."""
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")
        return self.versions[version]

# Test data
TEST_VERSIONS = {
    "1.0.0": ConfigVersionInfo("1.0.0", ConfigVersionStatus.CURRENT, "Current version"),
    "0.9.0": ConfigVersionInfo("0.9.0", ConfigVersionStatus.DEPRECATED, "Deprecated version"),
    "0.8.0": ConfigVersionInfo("0.8.0", ConfigVersionStatus.UNSUPPORTED, "Unsupported version"),
}

TEST_MIGRATIONS = {
    ("0.9.0", "1.0.0"): lambda config: {
        **config,
        "config_version": "1.0.0",
        "new_field": "migrated_value"
    }
}


class TestConfigVersioner:
    """Tests for the ConfigVersioner class."""

    @pytest.fixture
    def versioner(self):
        """Create a test ConfigVersioner instance."""
        return ConfigVersioner(
            versions=TEST_VERSIONS,
            migrations=TEST_MIGRATIONS
        )

    def test_check_version_current(self, versioner):
        """Test checking a current version."""
        status = versioner.check_version("1.0.0")
        assert status == ConfigVersionStatus.CURRENT

    def test_check_version_deprecated(self, versioner):
        """Test checking a deprecated version."""
        with pytest.warns(DeprecationWarning, match="Version 0.9.0 is deprecated"):
            status = versioner.check_version("0.9.0")
        assert status == ConfigVersionStatus.DEPRECATED

    def test_check_version_unsupported(self, versioner):
        """Test checking an unsupported version."""
        with pytest.raises(ValueError, match="Unsupported version: 0.7.0"):
            versioner.check_version("0.7.0")

    def test_migrate_config_no_migration_needed(self, versioner):
        """Test migrating a config that's already at the latest version."""
        config = {"config_version": "1.0.0", "key": "value"}
        migrated = versioner.migrate_config(config)
        assert migrated == config

    def test_migrate_config_with_migration(self, versioner):
        """Test migrating a config that needs migration."""
        old_config = {"config_version": "0.9.0", "key": "value"}
        migrated = versioner.migrate_config(old_config)
        
        assert migrated["config_version"] == "1.0.0"
        assert migrated["new_field"] == "migrated_value"
        assert migrated["key"] == "value"

    def test_migrate_config_unsupported_version(self, versioner):
        """Test migrating a config with an unsupported version."""
        with pytest.raises(ValueError, match="Unsupported"):
            versioner.migrate_config({"config_version": "0.7.0"})

    def test_migrate_config_missing_migration(self, versioner):
        """Test migrating a config when a migration is missing."""
        # Create a versioner with no migrations
        versioner = ConfigVersioner(versions=TEST_VERSIONS, migrations={})
        
        with pytest.raises(ValueError, match="No migration available from 0.9.0 to 1.0.0"):
            versioner.migrate_config({"config_version": "0.9.0"})

    def test_get_version_info(self, versioner):
        """Test getting version information."""
        info = versioner.get_version_info("1.0.0")
        assert info.version == "1.0.0"
        assert info.status == ConfigVersionStatus.CURRENT
        assert "Current version" in info.message

    def test_get_version_info_not_found(self, versioner):
        """Test getting version information for a non-existent version."""
        with pytest.raises(ValueError, match="Unknown version"):
            versioner.get_version_info("0.0.1")
