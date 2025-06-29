"""
Tests for the configuration versioning system.
"""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

# Import the actual implementation from the module where it's defined
from medvllm.medical.config.versioning.config_versioner import (
    ConfigVersioner,
    ConfigVersionInfo,
    ConfigVersionStatus,
    _migrate_090_to_100,
)


class TestConfigVersioner:
    """Test cases for ConfigVersioner class."""

    @pytest.fixture
    def versioner(self) -> ConfigVersioner:
        """Fixture that provides a ConfigVersioner instance."""
        return ConfigVersioner()

    @pytest.mark.parametrize("version", ["1.0.0", "0.9.0", "0.8.0"])
    def test_check_version_status(
        self, versioner: ConfigVersioner, version: str
    ) -> None:
        """Test checking version status."""
        version_info = versioner.get_version_info(version)
        assert version_info is not None
        # Just verify the version info exists and has the expected version
        assert version_info.version == version

    def test_validate_version_compatible_current(
        self, versioner: ConfigVersioner
    ) -> None:
        """Test validation of current version."""
        # Should not raise
        versioner.check_version_compatibility("1.0.0")

    def test_validate_version_compatible_deprecated(
        self, versioner: ConfigVersioner
    ) -> None:
        """Test validation of deprecated version."""
        # Should issue a deprecation warning
        with pytest.warns(DeprecationWarning, match="deprecated"):
            versioner.check_version_compatibility("0.9.0")

    def test_validate_version_compatible_unsupported(
        self, versioner: ConfigVersioner
    ) -> None:
        """Test validation of unsupported version."""
        with pytest.raises(ValueError, match="Unsupported version: 0.8.0"):
            versioner.check_version_compatibility("0.8.0")

    def test_validate_version_compatible_future(
        self, versioner: ConfigVersioner
    ) -> None:
        """Test validation of future version."""
        with pytest.raises(ValueError, match="Unsupported version: 2.0.0"):
            versioner.check_version_compatibility("2.0.0")

    def test_validate_version_compatible_invalid(
        self, versioner: ConfigVersioner
    ) -> None:
        """Test validation of invalid version."""
        with pytest.raises(ValueError, match="Unsupported version: invalid"):
            versioner.check_version_compatibility("invalid")

    def test_get_version_info(self, versioner: ConfigVersioner) -> None:
        """Test getting version information."""
        # Test getting known versions
        for version in ["1.0.0", "0.9.0", "0.8.0"]:
            info = versioner.get_version_info(version)
            assert info is not None
            assert info.version == version
            # Check that status is one of the valid enum values
            assert info.status in [
                ConfigVersionStatus.CURRENT,
                ConfigVersionStatus.DEPRECATED,
                ConfigVersionStatus.UNSUPPORTED,
            ]

        # Test getting non-existent version - should raise ValueError
        with pytest.raises(ValueError, match="Unknown version"):
            versioner.get_version_info("0.0.1")

    def test_migrate_090_to_100(self) -> None:
        """Test the migration from 0.9.0 to 1.0.0."""
        # Create a test config in the old format
        old_config = {"model_path": "/path/to/model", "other_setting": "value"}

        # Perform the migration
        migrated_config = _migrate_090_to_100(old_config)

        # Verify the migration
        assert "model_name_or_path" in migrated_config
        assert migrated_config["model_name_or_path"] == "/path/to/model"
        assert "model_path" not in migrated_config  # Old field should be removed
        assert (
            migrated_config["other_setting"] == "value"
        )  # Other settings should be preserved
