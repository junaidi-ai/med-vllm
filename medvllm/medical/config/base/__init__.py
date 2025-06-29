"""
Base configuration class for medical models.

This module contains the base configuration class that provides common
functionality for medical model configurations.
"""

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
    """

    config_version: str = field(default=CONFIG_VERSION, init=False)
    file_path: Optional[str] = None  # Path to the config file if loaded from disk

    def __post_init__(self):
        """Initialize the configuration and validate all parameters."""
        # Ensure compatibility with the current version
        self.ensure_compatibility()
        super().__post_init__()

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
            if hasattr(self, "medical_params") and isinstance(
                self.medical_params, dict
            ):
                for key, value in self.medical_params.items():
                    if not hasattr(self, key):
                        setattr(self, key, value)
                delattr(self, "medical_params")
            self.config_version = "1.0.0"
