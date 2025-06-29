"""
Type-related utility functions for configuration handling.

This module provides functions for working with Python type hints,
especially in the context of configuration validation and processing.
"""

import inspect
import typing
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Tuple, 
    Type, 
    TypeVar, 
    Union,
    get_args, 
    get_origin,
    _GenericAlias,
)

# Type variable for generic types
T = TypeVar('T')

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
    """Get the key and value types from a Dict[K, V] or dict[K, V] type hint.
    
    Args:
        t: The dict type
        
    Returns:
        A tuple of (key_type, value_type) if the input is a dict type, None otherwise
    """
    if not is_dict_type(t):
        return None
        
    args = get_args(t)
    if not args:
        return (Any, Any)
    if len(args) == 2:
        return (args[0], args[1])
    return (args[0], Any)

def is_subclass_or_instance(obj: Any, class_or_tuple: Union[type, Tuple[type, ...]]) -> bool:
    """Safely check if an object is a subclass or instance of a class.
    
    This works with typing types like List[str] and Union types.
    
    Args:
        obj: The object or class to check
        class_or_tuple: A class or tuple of classes to check against
        
    Returns:
        bool: True if obj is a subclass or instance of class_or_tuple
    """
    try:
        if inspect.isclass(obj):
            return issubclass(obj, class_or_tuple)
        return isinstance(obj, class_or_tuple)
    except TypeError:
        # Handle typing types and other special cases
        if hasattr(obj, '__origin__'):
            if obj.__origin__ is Union:
                return any(is_subclass_or_instance(arg, class_or_tuple) 
                          for arg in getattr(obj, '__args__', ()))
            return is_subclass_or_instance(obj.__origin__, class_or_tuple)
        return False
