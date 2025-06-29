"""
Protocols and interfaces for configuration validation.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConfigValidator(Protocol):
    """Protocol for configuration validation."""

    def validate(self, config: Any) -> bool:
        """Validate the configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if the configuration is valid, False otherwise
        """
        ...


@runtime_checkable
class ConfigSerializable(Protocol):
    """Protocol for serializable configuration objects."""

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        Returns:
            dict: Dictionary representation of the configuration
        """
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "ConfigSerializable":
        """Create a configuration from a dictionary.

        Args:
            data: Dictionary containing configuration data

        Returns:
            ConfigSerializable: A new configuration instance
        """
        ...
