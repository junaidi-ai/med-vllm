"""
Medical package for medical-specific functionality.
"""

# Explicitly import config to ensure it's registered as a submodule
from . import config

# Core imports
from .config import MedicalModelConfig

__all__ = ["MedicalModelConfig", "config"]
