"""
Medical model configuration module.

This package provides configuration management for medical models, including:
- MedicalModelConfig: Main configuration class
- Type definitions and enums
- Validation and serialization utilities
- Versioning support
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union

# Re-export base configuration
from .base import BaseMedicalConfig

# Import from constants
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

# Import main configuration class
from .medical_config import MedicalModelConfig

# Import serialization
from .serialization import ConfigSerializer, JSONSerializer, YAMLSerializer

# Import enums from types_
from .types_ import (
    AnatomicalRegion,
    DocumentType,
    EntityType,
    ImagingModality,
    MedicalSpecialty,
    RegulatoryStandard,
)

# Import validation
from .validation import MedicalConfigValidator

# Import versioning utilities
from .versioning.config_versioner import (
    ConfigVersioner,
    ConfigVersionInfo,
    ConfigVersionStatus,
)

# Define public API
__all__ = [
    # Main configuration classes
    "MedicalModelConfig",
    "BaseMedicalConfig",
    # Enums
    "AnatomicalRegion",
    "DocumentType",
    "ImagingModality",
    "MedicalSpecialty",
    "RegulatoryStandard",
    "EntityType",
    "ConfigVersionStatus",
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
    "JSONSerializer",
    "YAMLSerializer",
    "ConfigValidator",
    "MedicalConfigValidator",
    "ConfigVersioner",
    "ConfigVersionInfo",
]
