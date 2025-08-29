"""
Serialization and deserialization utilities for medical model configurations.

This module provides a unified interface for serializing and deserializing
medical model configurations to/from various formats like JSON and YAML.
"""

import warnings

# Fully-qualified module name for the submodule
_YAML_SERIALIZER_FQMN = "medvllm.medical.config.serialization.yaml_serializer"
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

from ..base import BaseMedicalConfig
from .config_serializer import ConfigSerializer
from .json_serializer import JSONSerializer

# Type variable for generic configuration types
T = TypeVar("T", bound=BaseMedicalConfig)

"""Expose yaml_serializer submodule consistently, even without PyYAML.

When PyYAML (or the submodule) is unavailable, provide a stub module with a
YAMLSerializer class whose methods raise ImportError. Tests expect the
`yaml_serializer` attribute to always exist on this package.
"""

try:
    # Expose real submodule if importable
    from . import yaml_serializer as yaml_serializer  # type: ignore  # noqa: F401
    from .yaml_serializer import PYYAML_AVAILABLE
    from .yaml_serializer import YAMLSerializer as YAML

    YAML_AVAILABLE = PYYAML_AVAILABLE
    YAMLSerializer = YAML if YAML_AVAILABLE else None
    # Ensure attribute exists on the package module object
    import sys as _sys

    globals()["yaml_serializer"] = yaml_serializer
    # Ensure the submodule is registered in sys.modules under its FQMN
    if _YAML_SERIALIZER_FQMN not in _sys.modules:
        _sys.modules[_YAML_SERIALIZER_FQMN] = yaml_serializer
except ImportError:
    # Create a stub module with a guard-raising YAMLSerializer
    import types as _types
    import sys as _sys

    yaml_serializer = _types.ModuleType(_YAML_SERIALIZER_FQMN)

    class _StubYAMLSerializer:  # pragma: no cover - simple guard class
        @classmethod
        def to_yaml(cls, *args, **kwargs):
            raise ImportError("PyYAML is required for YAML serialization")

        @classmethod
        def from_yaml(cls, *args, **kwargs):
            raise ImportError("PyYAML is required for YAML deserialization")

    # Public API surface on the stub
    setattr(yaml_serializer, "YAMLSerializer", _StubYAMLSerializer)
    setattr(yaml_serializer, "PYYAML_AVAILABLE", False)

    # Register stub in sys.modules so `from ... import yaml_serializer` works
    _sys.modules[_YAML_SERIALIZER_FQMN] = yaml_serializer
    # Also attach attribute on the package module object for `from pkg import yaml_serializer`
    globals()["yaml_serializer"] = yaml_serializer

    YAML_AVAILABLE = False
    YAMLSerializer = _StubYAMLSerializer

yaml_warning = (
    "PyYAML is not installed. YAML serialization will not be available. "
    "Install with: pip install pyyaml"
)

if not YAML_AVAILABLE:
    warnings.warn(yaml_warning, ImportWarning, stacklevel=2)


def __getattr__(name: str):  # PEP 562 dynamic attributes for robustness
    if name == "yaml_serializer":
        # Lazily ensure yaml_serializer attribute exists even if removed by tests
        try:
            from . import yaml_serializer as _ys  # type: ignore

            globals()["yaml_serializer"] = _ys
            import sys as _sys

            if _YAML_SERIALIZER_FQMN not in _sys.modules:
                _sys.modules[_YAML_SERIALIZER_FQMN] = _ys
            return _ys
        except Exception:
            # Fallback to stub
            import types as _types
            import sys as _sys

            _ys = _types.ModuleType(_YAML_SERIALIZER_FQMN)

            class _StubYAMLSerializer:  # pragma: no cover
                @classmethod
                def to_yaml(cls, *args, **kwargs):
                    raise ImportError("PyYAML is required for YAML serialization")

                @classmethod
                def from_yaml(cls, *args, **kwargs):
                    raise ImportError("PyYAML is required for YAML deserialization")

            setattr(_ys, "YAMLSerializer", _StubYAMLSerializer)
            setattr(_ys, "PYYAML_AVAILABLE", False)
            _sys.modules[_YAML_SERIALIZER_FQMN] = _ys
            globals()["yaml_serializer"] = _ys
            return _ys
    raise AttributeError(name)


def save_config(
    config: Union[BaseMedicalConfig, Dict[str, Any]],
    file_path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Save configuration to a file.

    The format is determined by the file extension if not specified.

    Args:
        config: Configuration to save
        file_path: Path to save the configuration to
        format: Optional format ('json' or 'yaml'). If None, inferred
               from file extension.
        **kwargs: Additional arguments to pass to the serializer

    Raises:
        ValueError: If the format is not supported or cannot be inferred
        ImportError: If the required dependencies are not installed
    """
    file_path_str = str(file_path)
    if format is None:
        suffix = Path(file_path_str).suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix in (".yaml", ".yml"):
            format = "yaml"
        else:
            raise ValueError(
                "Could not infer format from file extension. "
                "Please specify format='json' or format='yaml'"
            )

    format = format.lower()
    if format == "json":
        JSONSerializer.save_to_file(config, file_path_str, **kwargs)
    elif format == "yaml":
        if YAMLSerializer is None:
            raise ImportError(
                "PyYAML is required for YAML serialization. " "Install with: pip install pyyaml"
            )
        YAMLSerializer.to_yaml(config, file_path_str, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_config(
    file_path: Union[str, Path],
    config_class: Optional[Type[T]] = None,
    format: Optional[str] = None,
    **kwargs: Any,
) -> Union[Dict[str, Any], T]:
    """Load configuration from a file.

    The format is determined by the file extension if not specified.

    Args:
        file_path: Path to the configuration file
        config_class: Configuration class to instantiate
        format: Optional format ('json' or 'yaml'). If None, inferred
               from file extension.
        **kwargs: Additional arguments to pass to the deserializer

    Returns:
        An instance of the specified configuration class

    Raises:
        ValueError: If the format is not supported or cannot be inferred,
                   or if the deserialized data is invalid
        ImportError: If the required dependencies are not installed
    """
    # Determine format from file extension if not specified
    file_path_str = str(file_path)
    if format is None:
        suffix = Path(file_path_str).suffix.lower()
        if suffix == ".json":
            file_format = "json"
        elif suffix in (".yaml", ".yml"):
            file_format = "yaml"
        else:
            raise ValueError(
                "Could not infer format from file extension. "
                "Please specify format='json' or format='yaml'"
            )
    else:
        file_format = format.lower()

    # Validate format
    if file_format not in ("json", "yaml"):
        raise ValueError(f"Unsupported format: {file_format}")

    # Load and process the configuration based on format
    if file_format == "json":
        return _load_json_config(file_path_str, config_class, **kwargs)
    else:
        return _load_yaml_config(file_path_str, config_class, **kwargs)


def _load_json_config(
    file_path: str, config_class: Optional[Type[T]], **kwargs: Any
) -> Union[Dict[str, Any], T]:
    """Load configuration from a JSON file.

    Args:
        file_path: Path to the JSON configuration file
        config_class: Configuration class to instantiate
        **kwargs: Additional arguments to pass to the deserializer

    Returns:
        An instance of the specified configuration class

    Raises:
        ValueError: If the deserialized data is invalid
    """
    if config_class is None:
        # Return raw dictionary
        return JSONSerializer.deserialize(file_path, **kwargs)  # type: ignore[return-value]
    result = JSONSerializer.deserialize(file_path, config_class, **kwargs)
    return result


def _load_yaml_config(
    file_path: str, config_class: Optional[Type[T]], **kwargs: Any
) -> Union[Dict[str, Any], T]:
    """Load configuration from a YAML file.

    Args:
        file_path: Path to the YAML configuration file
        config_class: Configuration class to instantiate
        **kwargs: Additional arguments to pass to the deserializer

    Returns:
        An instance of the specified configuration class

    Raises:
        ImportError: If PyYAML is not installed
        ValueError: If the deserialized data is invalid
    """
    if YAMLSerializer is None:
        raise ImportError(
            "PyYAML is required for YAML deserialization. " "Install with: pip install pyyaml"
        )

    if config_class is None:
        return YAMLSerializer.from_yaml(file_path, None, **kwargs)  # type: ignore[return-value]
    return YAMLSerializer.from_yaml(file_path, config_class, **kwargs)


__all__ = [
    "ConfigSerializer",
    "JSONSerializer",
    "YAMLSerializer",
    "yaml_serializer",
    "save_config",
    "load_config",
]
