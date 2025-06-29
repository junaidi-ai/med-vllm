"""
Utility functions for medical model configurations.

This package contains various utility functions that are used throughout
the configuration system, including type conversion, path handling, and
other helper functions.
"""

from .type_utils import (
    is_optional_type,
    get_optional_type,
    is_union_type,
    get_union_types,
    is_list_type,
    get_list_item_type,
    is_dict_type,
    get_dict_types,
)
from .path_utils import (
    ensure_path,
    resolve_config_path,
    find_config_file,
)
from .misc import (
    deep_update,
    filter_none_values,
    get_nested_value,
    set_nested_value,
)

__all__ = [
    # Type utilities
    'is_optional_type',
    'get_optional_type',
    'is_union_type',
    'get_union_types',
    'is_list_type',
    'get_list_item_type',
    'is_dict_type',
    'get_dict_types',
    
    # Path utilities
    'ensure_path',
    'resolve_config_path',
    'find_config_file',
    
    # Miscellaneous utilities
    'deep_update',
    'filter_none_values',
    'get_nested_value',
    'set_nested_value',
]
