"""
Utility functions for type handling in configuration.
"""

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

def validate_model_path(path: Optional[PathLike]) -> Optional[str]:
    """Validate and normalize the model path.

    Args:
        path: Path to validate

    Returns:
        Normalized path as string or None if path is None
    """
    if path is None:
        return None
    
    path_str = str(path)
    if not path_str.strip():
        return None
        
    path_obj = Path(path_str).expanduser().resolve()
    return str(path_obj)

def ensure_list(value: Union[list, tuple, set, str, int, float, bool, None]) -> list:
    """Ensure the value is a list.
    
    Args:
        value: Value to convert to a list
        
    Returns:
        list: The value as a list
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]

def ensure_dict(value: Union[dict, list, tuple, None]) -> dict:
    """Ensure the value is a dictionary.
    
    Args:
        value: Value to convert to a dictionary
        
    Returns:
        dict: The value as a dictionary
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in value):
        return dict(value)
    raise ValueError(f"Cannot convert {type(value)} to dictionary")
