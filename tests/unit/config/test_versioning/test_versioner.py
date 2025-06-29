"""
Tests for the configuration versioning system.
"""

import pytest
from unittest.mock import MagicMock, patch

# Import the actual implementation
from medvllm.medical.config.versioning import ConfigVersioner, VersionStatus
from medvllm.medical.config.exceptions import VersionCompatibilityError


class TestConfigVersioner:
    """Test cases for ConfigVersioner class."""
    
    @pytest.fixture
    def versioner(self) -> ConfigVersioner:
        """Return a ConfigVersioner instance for testing."""
        return ConfigVersioner(current_version="1.0.0")
    
    @pytest.mark.parametrize("version,status", [
        ("1.0.0", VersionStatus.CURRENT),
        ("0.9.0", VersionStatus.OUTDATED),
        ("1.1.0", VersionStatus.FUTURE),
        ("invalid", VersionStatus.INVALID),
    ])
    def test_check_version_status(self, versioner: ConfigVersioner, version: str, status: VersionStatus) -> None:
        """Test checking version status."""
        assert versioner.check_version_status(version) == status
    
    def test_validate_version_compatible_current(self, versioner: ConfigVersioner) -> None:
        """Test validation of current version."""
        # Should not raise
        versioner.validate_version_compatible("1.0.0")
    
    def test_validate_version_compatible_outdated(self, versioner: ConfigVersioner) -> None:
        """Test validation of outdated version."""
        with pytest.raises(VersionCompatibilityError, match="outdated"):
            versioner.validate_version_compatible("0.9.0")
    
    def test_validate_version_compatible_future(self, versioner: ConfigVersioner) -> None:
        """Test validation of future version."""
        with pytest.raises(VersionCompatibilityError, match="newer"):
            versioner.validate_version_compatible("2.0.0")
    
    def test_validate_version_compatible_invalid(self, versioner: ConfigVersioner) -> None:
        """Test validation of invalid version."""
        with pytest.raises(VersionCompatibilityError, match="invalid"):
            versioner.validate_version_compatible("invalid")
    
    def test_register_migration(self, versioner: ConfigVersioner) -> None:
        """Test registering a migration."""
        # Given
        def mock_migration(config: dict) -> dict:
            config["migrated"] = True
            return config
            
        # When
        versioner.register_migration("0.9.0", "1.0.0", mock_migration)
        
        # Then
        assert ("0.9.0", "1.0.0") in versioner._migrations
    
    def test_migrate(self, versioner: ConfigVersioner) -> None:
        """Test migrating a configuration."""
        # Given
        def mock_migration(config: dict) -> dict:
            config["migrated"] = True
            return config
            
        versioner.register_migration("0.9.0", "1.0.0", mock_migration)
        config = {"version": "0.9.0"}
        
        # When
        migrated = versioner.migrate(config)
        
        # Then
        assert migrated["version"] == "1.0.0"
        assert migrated["migrated"] is True
    
    def test_migrate_no_migration_needed(self, versioner: ConfigVersioner) -> None:
        """Test migrating a configuration that's already current."""
        # Given
        config = {"version": "1.0.0"}
        
        # When
        migrated = versioner.migrate(config)
        
        # Then - should be the same object, no changes
        assert migrated is config
    
    def test_migrate_no_version(self, versioner: ConfigVersioner) -> None:
        """Test migrating a configuration with no version."""
        # Given
        config = {}
        
        # When/Then
        with pytest.raises(ValueError, match="No version specified"):
            versioner.migrate(config)
    
    def test_migrate_no_path(self, versioner: ConfigVersioner) -> None:
        """Test migrating when no migration path exists."""
        # Given
        config = {"version": "0.8.0"}
        
        # When/Then
        with pytest.raises(ValueError, match="No migration path"):
            versioner.migrate(config)
