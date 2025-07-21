"""Utility functions for medical model adapters.

This package provides various utility functions and classes for working with medical
language models, including attention mechanisms, layer operations, and model utilities.
"""

from . import attention_utils, layer_utils
from .attention_utils import (
    apply_attention,
    combine_heads,
    get_attention_mask,
)
from .attention_utils import (
    get_extended_attention_mask as get_extended_attention_mask_from_utils,
)
from .attention_utils import (
    split_heads,
)
from .layer_utils import (
    create_initializer,
    create_position_ids_from_input_ids,
    create_sinusoidal_positional_embedding,
    get_activation_fn,
    get_device_map,
    get_parameter_count,
    get_parameter_device,
    get_parameter_dtype,
)

# Re-export all symbols
__all__ = [
    # Modules
    "attention_utils",
    "layer_utils",
    # Attention utilities
    "apply_attention",
    "combine_heads",
    "get_attention_mask",
    "get_extended_attention_mask_from_utils",
    "split_heads",
    # Layer utilities
    "create_initializer",
    "create_position_ids_from_input_ids",
    "create_sinusoidal_positional_embedding",
    "get_activation_fn",
    "get_device_map",
    "get_parameter_count",
    "get_parameter_dtype",
    "get_parameter_device",
]
