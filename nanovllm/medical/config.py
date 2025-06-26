"""
Legacy configuration module for backward compatibility.

This module provides backward compatibility for code that imports from
`nanovllm.medical.config`. New code should import from the refactored
modules in `nanovllm.medical.config` directly.
"""

import warnings
from typing import Any, Dict, Optional, Union

from .config.medical_config import MedicalModelConfig

# Show deprecation warning
warnings.warn(
    "The module 'nanovllm.medical.config' is deprecated and will be removed in a future version. "
    "Please update imports to use 'nanovllm.medical.config.medical_config' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export the main class for backward compatibility
__all__ = ["MedicalModelConfig"]
