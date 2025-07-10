"""
Type-related utility functions for configuration handling.

This module provides functions for working with Python type hints,
especially in the context of configuration validation and processing.
"""

import typing
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

# Type variable for generic types
T = TypeVar("T")


def is_optional_type(t: type) -> bool:
    """Check if a type is Optional[T] (which is Union[T, None]).

    Args:
        t: The type to check

    Returns:
        bool: True if the type is Optional[T], False otherwise
    """
    if t is type(None):  # noqa: E721
        return True

    origin = get_origin(t)
    if origin is not Union:
        return False

    return type(None) in get_args(t)


def get_optional_type(t: type) -> Optional[type]:
    """Get the inner type from an Optional[T] type hint.

    Args:
        t: The type to check (expected to be Optional[T])

    Returns:
        The inner type T if the input is Optional[T], None otherwise
    """
    if not is_optional_type(t):
        return None

    args = get_args(t)
    if not args:
        return None

    # Return the first non-None type
    for arg in args:
        if arg is not type(None):  # noqa: E721
            return arg
    return None


def is_union_type(t: type) -> bool:
    """Check if a type is a Union type.

    Args:
        t: The type to check

    Returns:
        bool: True if the type is a Union, False otherwise
    """
    if is_optional_type(t):
        return False

    origin = get_origin(t)
    return origin is Union


def get_union_types(t: type) -> Tuple[type, ...]:
    """Get the types from a Union type hint.

    Args:
        t: The Union type

    Returns:
        A tuple of types in the Union

    Raises:
        TypeError: If the input is not a Union type
    """
    if not is_union_type(t):
        raise TypeError(f"Expected a Union type, got {t}")
    return get_args(t)


def is_list_type(t: type) -> bool:
    """Check if a type is a List[T] or list[T] type.

    Args:
        t: The type to check

    Returns:
        bool: True if the type is a List[T] or list[T], False otherwise
    """
    origin = get_origin(t) or t
    return origin in (list, List, typing.List)


def get_list_item_type(t: type) -> Optional[type]:
    """Get the item type from a List[T] or list[T] type hint.

    Args:
        t: The list type

    Returns:
        The item type T if the input is List[T] or list[T], None otherwise
    """
    if not is_list_type(t):
        return None

    args = get_args(t)
    return args[0] if args else Any


def is_dict_type(t: type) -> bool:
    """Check if a type is a Dict[K, V] or dict[K, V] type.

    Args:
        t: The type to check

    Returns:
        bool: True if the type is a Dict[K, V] or dict[K, V], False otherwise
    """
    origin = get_origin(t) or t
    return origin in (dict, Dict, typing.Dict)


def get_dict_types(t: type) -> Optional[Tuple[type, type]]:
    """Get key and value types from a Dict[K, V] or dict[K, V] type hint.

    Args:
        t: The dict type

    Returns:
        A tuple of (key_type, value_type) if the input is a dict type,
        None otherwise
    """
    if not is_dict_type(t):
        return None

    args = get_args(t)
    if not args:
        return (Any, Any)
    if len(args) == 2:
        return (args[0], args[1])
    return (args[0], Any)


def is_basic_type(t: type) -> bool:
    """Check if a type is a basic Python type.

    Basic types include: int, float, bool, str, and NoneType.

    Args:
        t: The type to check

    Returns:
        bool: True if the type is a basic type, False otherwise
    """
    basic_types = (int, float, bool, str, type(None))
    return t in basic_types or t is Any or t is type(None)


def convert_string_to_type(value: str, target_type: type | None) -> Any:
    """Convert a string to the specified type.

    Args:
        value: The string value to convert
        target_type: The target type to convert to, or None to return the value as-is

    Returns:
        The converted value

    Raises:
        ValueError: If the conversion fails
    """
    if target_type is None or not isinstance(value, str):
        return value

    if target_type is str:
        return value
    elif target_type is int:
        return int(value)
    elif target_type is float:
        return float(value)
    elif target_type is bool:
        return value.lower() in ("true", "yes", "1")
    elif is_list_type(target_type):
        import json

        item_type = get_list_item_type(target_type)
        try:
            parsed_list = json.loads(value)
            if not isinstance(parsed_list, list):
                raise ValueError(
                    f"Expected a JSON array, got {type(parsed_list).__name__}"
                )
            return [
                convert_string_to_type(str(item), item_type) for item in parsed_list
            ]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for list conversion: {e}")
    elif is_dict_type(target_type):
        import json

        dict_types = get_dict_types(target_type)
        if dict_types is None:
            raise ValueError(
                f"Could not determine key and value types for dict type: {target_type}"
            )

        key_type, value_type = dict_types
        try:
            parsed_dict = json.loads(value)
            if not isinstance(parsed_dict, dict):
                raise ValueError(
                    f"Expected a JSON object, got {type(parsed_dict).__name__}"
                )
            return {
                convert_string_to_type(str(k), key_type): (
                    convert_string_to_type(str(v), value_type)
                )
                for k, v in parsed_dict.items()
            }
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for dict conversion: {e}")
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


def validate_type(value: Any, type_hint: type) -> bool:
    """Validate that a value matches a type hint.

    Args:
        value: The value to validate
        type_hint: The type hint to validate against

    Returns:
        bool: True if the value matches the type hint, False otherwise
    """
    # Handle None case first
    if value is None:
        is_optional = is_optional_type(type_hint)
        has_optional_type = get_optional_type(type_hint) is not None
        return type_hint is type(None) or (is_optional and has_optional_type)

    # Handle basic types
    if type_hint in (int, float, bool, str):
        return isinstance(value, type_hint)

    # Handle Any type
    if type_hint is Any:
        return True

    # Handle Optional types
    if is_optional_type(type_hint):
        inner_type = get_optional_type(type_hint)
        if inner_type is not None:  # Check if inner_type is not None before using it
            return validate_type(value, inner_type)
        return value is None

    # Handle Union types
    if is_union_type(type_hint):
        return any(validate_type(value, t) for t in get_union_types(type_hint))

    # Handle List types
    if is_list_type(type_hint):
        if not isinstance(value, (list, tuple)):
            return False
        item_type = get_list_item_type(type_hint)
        if item_type is None:
            return True  # If we can't determine the item type, accept any items
        return all(validate_type(item, item_type) for item in value)

    # Handle Dict types
    if is_dict_type(type_hint):
        if not isinstance(value, dict):
            return False
        dict_types = get_dict_types(type_hint)
        if dict_types is None:
            return True  # If we can't determine the key/value types, accept any dict
        key_type, value_type = dict_types
        return all(
            validate_type(k, key_type) and validate_type(v, value_type)
            for k, v in value.items()
        )

    # Handle other types using isinstance
    try:
        return isinstance(value, type_hint)
    except TypeError:
        # If isinstance fails, try to handle typing types
        if hasattr(type_hint, "__origin__"):
            if type_hint.__origin__ is Union and hasattr(type_hint, "__args__"):
                return any(validate_type(value, t) for t in type_hint.__args__)
            # Handle other typing types as needed
            return True  # Default to True for complex types we can't check
        return False


def is_subclass_or_instance(
    obj: Any, class_or_tuple: Union[type, Tuple[type, ...]]
) -> bool:
    """Safely check if an object is a subclass or instance of a class.

    This works with typing types like List[str] and Union types.

    Args:
        obj: The object or class to check
        class_or_tuple: A class or tuple of classes to check against

    Returns:
        bool: True if obj is a subclass or instance of class_or_tuple
    """
    try:
        if isinstance(obj, type):
            # For classes, use issubclass
            return issubclass(obj, class_or_tuple)
        else:
            # For instances, use isinstance
            return isinstance(obj, class_or_tuple)
    except (TypeError, AttributeError):
        # Handle cases where the check isn't possible (like with typing types)
        # For now, just return False to be safe
        return False
    if hasattr(obj, "__origin__"):
        if obj.__origin__ is Union:
            return any(
                is_subclass_or_instance(arg, class_or_tuple)
                for arg in getattr(obj, "__args__", ())
            )
        return is_subclass_or_instance(obj.__origin__, class_or_tuple)
    return False
