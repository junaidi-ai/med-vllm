"""
Type definitions and utilities for medical model configuration.

This module provides type hints and data structures for the medical model configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field

# Re-export types from submodules
from .enums import (
    AnatomicalRegion,
    DocumentType,
    EntityType,
    ImagingModality,
    MedicalSpecialty,
    RegulatoryStandard,
)
from .models import (
    ClinicalMetrics,
    DomainConfig,
    EntityLinkingConfig,
    MetricConfig,
    MetricRange,
    ModelConfig,
)
from .protocols import ConfigSerializable, ConfigValidator
from .utils import PathLike, ensure_dict, ensure_list, validate_model_path

# Type variables for generic types
T = TypeVar("T")
