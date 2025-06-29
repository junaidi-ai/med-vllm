"""
Medical model configuration module.

This package provides configuration management for medical models, including:
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union

# Import all submodules to ensure they are properly registered
from . import base
from . import constants
from . import models
from . import serialization
from . import types
from . import utils
from . import validation
from . import versioning

# Import base configuration
from .base import BaseMedicalConfig

# Import constants
from .constants import (
    DEFAULT_ANATOMICAL_REGIONS as ANATOMICAL_REGIONS,
    DEFAULT_ANATOMICAL_REGIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DOCUMENT_TYPES,
    DEFAULT_DOMAIN_ADAPTATION_LAMBDA,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_IMAGING_MODALITIES,
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
from .models.medical_config import MedicalModelConfig

# Import serialization
from .serialization import ConfigSerializer, JSONSerializer, YAMLSerializer

# Import types
from .types import (
    AnatomicalRegion,
    ClinicalMetrics,
    ConfigSerializable,
    ConfigValidator,
    DocumentType,
    DomainConfig,
    EntityLinkingConfig,
    EntityType,
    ImagingModality,
    MedicalSpecialty,
    MetricConfig,
    MetricRange,
    ModelConfig,
    RegulatoryStandard,
    validate_model_path,
)

# Import validation
from .validation import MedicalConfigValidator

# Import versioning utilities
from .versioning import ConfigVersioner, ConfigVersionInfo, ConfigVersionStatus

# Define public API
__all__ = [
    # Main configuration classes
    "MedicalModelConfig",
    "BaseMedicalConfig",
    # Type classes
    "ModelConfig",
    "DomainConfig",
    "EntityLinkingConfig",
    "ClinicalMetrics",
    "MetricConfig",
    "MetricRange",
    # Enums
    "AnatomicalRegion",
    "DocumentType",
    "ImagingModality",
    "MedicalSpecialty",
    "EntityType",
    "RegulatoryStandard",
    # Constants
    "DEFAULT_MODEL_TYPE",
    "DEFAULT_MAX_SEQ_LENGTH",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NER_THRESHOLD",
    "DEFAULT_MAX_ENTITY_SPAN",
    "DEFAULT_UNCERTAINTY_THRESHOLD",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_DOMAIN_ADAPTATION_LAMBDA",
    "SUPPORTED_MODEL_TYPES",
    "DEFAULT_MEDICAL_SPECIALTIES",
    "DEFAULT_ANATOMICAL_REGIONS",
    "DEFAULT_IMAGING_MODALITIES",
    "DEFAULT_ENTITY_TYPES",
    "DEFAULT_DOCUMENT_TYPES",
    "DEFAULT_SECTION_HEADERS",
    "DEFAULT_REGULATORY_STANDARDS",
    # Validation
    "MedicalConfigValidator",
    "ConfigValidator",
    "ConfigSerializable",
    # Versioning
    "ConfigVersioner",
    "ConfigVersionInfo",
    "ConfigVersionStatus",
    # Serialization
    "ConfigSerializer",
    "JSONSerializer",
    "YAMLSerializer",
    # Utils
    "validate_model_path",
]
