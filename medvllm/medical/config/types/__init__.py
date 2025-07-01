"""Type definitions and utilities for medical model configuration.

This module provides type hints and data structures for the medical model
configuration system. It includes enums, models, protocols, and utilities
that define the structure and validation rules for medical model
configurations.
"""

from __future__ import annotations

from typing import TypeVar

# Import only what's needed from pydantic
from pydantic import BaseModel, Field  # noqa: F401 (re-exported)

# Re-export enums
from .enums import (
    AnatomicalRegion,
    DocumentType,
    EntityType,
    ImagingModality,
    MedicalSpecialty,
    RegulatoryStandard,
)

# Re-export models
from .models import (
    ClinicalMetrics,
    DomainConfig,
    EntityLinkingConfig,
    MetricConfig,
    MetricRange,
    ModelConfig,
)

# Re-export protocols
from .protocols import ConfigSerializable, ConfigValidator

# Re-export utils
from .utils import PathLike, ensure_dict, ensure_list, validate_model_path

# Re-export all public symbols for proper type checking and autocompletion
__all__ = [
    # From enums
    "AnatomicalRegion",
    "DocumentType",
    "EntityType",
    "ImagingModality",
    "MedicalSpecialty",
    "RegulatoryStandard",
    # From models
    "ClinicalMetrics",
    "DomainConfig",
    "EntityLinkingConfig",
    "MetricConfig",
    "MetricRange",
    "ModelConfig",
    # From protocols
    "ConfigSerializable",
    "ConfigValidator",
    # From utils
    "PathLike",
    "ensure_dict",
    "ensure_list",
    "validate_model_path",
    # From pydantic
    "BaseModel",
    "Field",
]

# Type variables for generic types
T = TypeVar("T")
