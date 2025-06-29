"""
Medical package for medical-specific functionality.
"""

# Core imports
from .config import MedicalModelConfig

# Explicitly import config to ensure it's registered as a submodule
from . import config

__all__ = [
    "MedicalModelConfig",
    "config"
]
