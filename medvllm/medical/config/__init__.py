"""
Medical configuration module.

This package provides configuration management for medical models, including:
- MedicalModelConfig: Main configuration class
- Type definitions and enums
- Validation and serialization utilities
- Versioning support
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union

# Re-export base configuration
from .base import BaseMedicalConfig

# Re-export constants
from .constants import (
    CONFIG_VERSION,
    DEFAULT_ANATOMICAL_REGIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DOCUMENT_TYPES,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_IMAGING_MODALITIES,
    DEFAULT_KNOWLEDGE_BASES,
    DEFAULT_MAX_ENTITY_SPAN,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MEDICAL_SPECIALTIES,
    DEFAULT_MODEL_TYPE,
    DEFAULT_NER_THRESHOLD,
    DEFAULT_REGULATORY_STANDARDS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SECTION_HEADERS,
    DEFAULT_UNCERTAINTY_THRESHOLD,
    SUPPORTED_MODEL_TYPES,
)

# Re-export main configuration class
from .medical_config import MedicalModelConfig

# Re-export serialization
from .serialization import ConfigSerializer

# Re-export types and enums
from .types import (
    AnatomicalRegion,
    ClinicalMetrics,
    DocumentType,
    DomainConfig,
    EntityLinkingConfig,
    EntityType,
    ImagingModality,
    MedicalSpecialty,
    MetricConfig,
    ModelConfig,
    RegulatoryStandard,
)

# Re-export validation
from .validation import MedicalConfigValidator

# Re-export versioning utilities
from .versioning import ConfigVersioner, ConfigVersionInfo, ConfigVersionStatus

# Define public API
__all__ = [
    # Main configuration class
    "MedicalModelConfig",
    # Base classes
    "BaseMedicalConfig",
    "ModelConfig",
    # Enums
    "AnatomicalRegion",
    "DocumentType",
    "EntityType",
    "ImagingModality",
    "MedicalSpecialty",
    "RegulatoryStandard",
    "ConfigVersionStatus",
    # Type definitions
    "ClinicalMetrics",
    "DomainConfig",
    "EntityLinkingConfig",
    "MetricConfig",
    # Constants
    "CONFIG_VERSION",
    "DEFAULT_ANATOMICAL_REGIONS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DOCUMENT_TYPES",
    "DEFAULT_ENTITY_TYPES",
    "DEFAULT_IMAGING_MODALITIES",
    "DEFAULT_KNOWLEDGE_BASES",
    "DEFAULT_MAX_ENTITY_SPAN",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_SEQ_LENGTH",
    "DEFAULT_MEDICAL_SPECIALTIES",
    "DEFAULT_MODEL_TYPE",
    "DEFAULT_NER_THRESHOLD",
    "DEFAULT_REGULATORY_STANDARDS",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_SECTION_HEADERS",
    "DEFAULT_UNCERTAINTY_THRESHOLD",
    "SUPPORTED_MODEL_TYPES",
    # Utilities
    "ConfigSerializer",
    "ConfigVersioner",
    "ConfigVersionInfo",
    "MedicalConfigValidator",
]
