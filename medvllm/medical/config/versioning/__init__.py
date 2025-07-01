"""
Versioning utilities for medical model configurations.

This module provides version management for configuration files,
including version checking, migration, and compatibility handling.
"""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel


class ConfigVersionStatus(str, Enum):
    """Status of a configuration version."""

    CURRENT = "current"
    DEPRECATED = "deprecated"
    UNSUPPORTED = "unsupported"


class ConfigVersionInfo(BaseModel):
    """Information about a configuration version."""

    version: str
    status: ConfigVersionStatus
    message: str = ""


class ConfigVersioner:
    """Manages configuration versions and their status."""

    VERSIONS: Dict[str, ConfigVersionInfo] = {
        "1.0.0": ConfigVersionInfo(
            version="1.0.0",
            status=ConfigVersionStatus.CURRENT,
            message="Stable release of medical configuration",
        ),
        "0.1.0": ConfigVersionInfo(
            version="0.1.0",
            status=ConfigVersionStatus.DEPRECATED,
            message=(
                "Initial release of medical configuration. " "Please upgrade to 1.0.0"
            ),
        ),
    }

    @classmethod
    def get_version_info(cls, version: str) -> Optional[ConfigVersionInfo]:
        """Get information about a specific version."""
        return cls.VERSIONS.get(version)

    @classmethod
    def check_version_compatibility(cls, version: str) -> bool:
        """Check version compatibility and issue warnings if deprecated."""
        version_info = cls.get_version_info(version)
        if not version_info:
            raise ValueError(f"Unsupported configuration version: {version}")

        if version_info.status == ConfigVersionStatus.DEPRECATED:
            import warnings

            warnings.warn(
                f"Configuration version {version} is deprecated. "
                f"{version_info.message}",
                DeprecationWarning,
                stacklevel=2,
            )
        elif version_info.status == ConfigVersionStatus.UNSUPPORTED:
            raise ValueError(
                f"Configuration version {version} is no longer "
                f"supported. {version_info.message}"
            )

        return True
