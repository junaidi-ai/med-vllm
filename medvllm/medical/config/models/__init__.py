"""Medical model configuration models.

This package contains Pydantic models and schemas for validating
and documenting the medical model configuration.
"""

from .schema import MedicalModelConfigSchema, ModelType
from .medical_config import MedicalModelConfig

__all__ = ["MedicalModelConfig", "MedicalModelConfigSchema", "ModelType"]
