"""
Miscellaneous utility functions for configuration handling.

This module contains various utility functions that don't fit into
the other utility categories but are still useful for configuration processing.
"""

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from pydantic import BaseModel

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def deep_update(dest: Dict[K, V], src: Mapping[K, V]) -> Dict[K, V]:
    """Recursively update a dictionary with another dictionary.

    This is similar to dict.update() but handles nested dictionaries
    by updating them recursively rather than replacing them.

    Args:
        dest: The dictionary to update
        src: The dictionary to merge into dest

    Returns:
        The updated dictionary (same object as dest)
    """
    for k, v in src.items():
        if isinstance(v, Mapping):
            dest[k] = deep_update(dest.get(k, {}), v)  # type: ignore
        else:
            dest[k] = v  # type: ignore
    return dest


def filter_none_values(data: Dict[K, Optional[V]]) -> Dict[K, V]:
    """Remove None values from a dictionary.

    Args:
        data: The dictionary to filter

    Returns:
        A new dictionary with None values removed
    """
    return {k: v for k, v in data.items() if v is not None}


def get_nested_value(
    data: Union[Dict[str, Any], BaseModel],
    key_path: Union[str, Sequence[str]],
    default: Any = None,
) -> Any:
    """Get a value from a nested dictionary.

    Supports dot notation or path segments for nested access.

    Args:
        data: The dictionary or Pydantic model to get the value from
        key_path: Either a dot-separated string or a list/tuple of keys
        default: The default value to return if the key is not found

    Returns:
        The value at the specified path, or the default if not found
    """
    if isinstance(data, BaseModel):
        data = data.dict()

    if isinstance(key_path, str):
        keys = key_path.split(".")
    else:
        keys = list(key_path)

    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def set_nested_value(
    data: Dict[str, Any],
    key_path: Union[str, Sequence[str]],
    value: Any,
    create_missing: bool = True,
) -> None:
    """Set a value in a nested dictionary using dot notation or path segments.

    Args:
        data: The dictionary to update
        key_path: Either a dot-separated string or a list/tuple of keys
        value: The value to set
        create_missing: If True, create missing intermediate dictionaries

    Raises:
        KeyError: If create_missing is False and a key in the path
                 doesn't exist
    """
    if isinstance(key_path, str):
        keys = key_path.split(".")
    else:
        keys = list(key_path)

    current = data
    for i, key in enumerate(keys[:-1]):
        if key not in current:
            if not create_missing:
                raise KeyError(
                    f"Key '{key}' not found in path '{'.'.join(keys[:i+1])}'"
                )
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def to_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable format.

    Handles common types including Pydantic models, dataclasses, enums, etc.

    Args:
        obj: The object to convert

    Returns:
        A JSON-serializable representation of the object
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle Pydantic models
    if hasattr(obj, "dict"):
        return obj.dict()

    # Handle dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses

        return {
            f.name: to_serializable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }

    # Handle enums
    if isinstance(obj, type) and issubclass(obj, type):
        return obj.__name__
    if (
        hasattr(obj, "value")
        and hasattr(obj, "__class__")
        and hasattr(obj.__class__, "_member_names_")
    ):
        return obj.value

    # Handle sequences
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(item) for item in obj]

    # Handle mappings
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}

    # Handle other objects with __dict__
    if hasattr(obj, "__dict__"):
        return {
            k: to_serializable(v) for k, v in vars(obj).items() if not k.startswith("_")
        }

    # Fallback to string representation
    return str(obj)
