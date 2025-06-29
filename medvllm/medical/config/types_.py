"""
Type definitions and utilities for medical model configuration.

This module provides type hints and data structures for the medical model configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field

# Type variables for generic types
T = TypeVar("T")
PathLike = Union[str, Path]


# Enums for type safety
class MedicalSpecialty(str, Enum):
    """Enumeration of medical specialties."""

    FAMILY_MEDICINE = "family_medicine"
    INTERNAL_MEDICINE = "internal_medicine"
    PEDIATRICS = "pediatrics"
    # Add other specialties as needed


class AnatomicalRegion(str, Enum):
    """Enumeration of anatomical regions."""

    HEAD = "head"
    THORAX = "thorax"
    ABDOMEN = "abdomen"
    # Add other regions as needed


class ImagingModality(str, Enum):
    """Enumeration of medical imaging modalities."""

    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    PET = "pet"
    MAMMOGRAPHY = "mammography"
    FLUOROSCOPY = "fluoroscopy"
    ANGIOGRAPHY = "angiography"


class EntityType(str, Enum):
    """Types of medical entities."""

    DISEASE = "DISEASE"
    SYMPTOM = "SYMPTOM"
    TREATMENT = "TREATMENT"
    MEDICATION = "MEDICATION"
    LAB_TEST = "LAB_TEST"
    ANATOMY = "ANATOMY"
    PROCEDURE = "PROCEDURE"
    FINDING = "FINDING"
    OBSERVATION = "OBSERVATION"
    FAMILY_HISTORY = "FAMILY_HISTORY"


class DocumentType(str, Enum):
    """Types of clinical documents."""

    CLINICAL_NOTES = "clinical_notes"
    RADIOLOGY_REPORTS = "radiology_reports"
    DISCHARGE_SUMMARIES = "discharge_summaries"
    PROGRESS_NOTES = "progress_notes"
    SURGICAL_REPORTS = "surgical_reports"
    PATHOLOGY_REPORTS = "pathology_reports"


class RegulatoryStandard(str, Enum):
    """Regulatory compliance standards."""

    HIPAA = "hipaa"
    GDPR = "gdpr"
    HL7 = "hl7"
    FDA_510K = "fda_510k"
    CE_MARK = "ce_mark"


# Typed dictionaries for configuration
class MetricRange(TypedDict):
    """Range for metric values with optional bounds."""

    min: float
    max: float
    unit: str


class MetricConfig(TypedDict, total=False):
    """Configuration for a single metric."""

    description: str
    unit: str
    normal_range: MetricRange
    critical_range: Optional[MetricRange]
    category: str
    higher_is_worse: bool
    required: bool


class ClinicalMetrics(TypedDict):
    """Configuration for clinical metrics."""

    vital_signs: Dict[str, MetricConfig]
    lab_tests: Dict[str, MetricConfig]
    scores: Dict[str, MetricConfig]


class DomainConfig(TypedDict):
    """Configuration for domain-specific settings."""

    domain_adaptation: bool
    domain_adaptation_lambda: float
    domain_specific_vocab: Optional[Dict[str, List[str]]]


class EntityLinkingConfig(TypedDict):
    """Configuration for entity linking."""

    enabled: bool
    knowledge_bases: List[str]
    confidence_threshold: float


class ModelConfig(BaseModel):
    """Base model configuration."""

    model_name: str = Field(..., description="Name or path of the model")
    model_type: str = Field(..., description="Type of the model architecture")
    max_sequence_length: int = Field(
        default=512, description="Maximum sequence length for the model"
    )
    do_lower_case: bool = Field(
        default=True, description="Whether to lowercase the input"
    )


# Protocol for configuration validation
@runtime_checkable
class ConfigValidator(Protocol):
    """Protocol for configuration validation."""

    def validate(self, config: Any) -> bool:
        """Validate the configuration."""
        ...


def validate_model_path(path: Optional[PathLike]) -> Optional[str]:
    """Validate and normalize the model path.

    Args:
        path: Path to validate

    Returns:
        Normalized path as string or None if path is None
    """
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())
