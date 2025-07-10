"""
Schema definitions for medical model configuration.

This module contains Pydantic models and schemas for validating
and documenting the medical model configuration.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medvllm.medical.config.types.models import (
    ClinicalMetrics,
    DomainConfig,
    EntityLinkingConfig,
    MetricConfig,
)

# Type variables for generics
T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)


class ModelType(str, Enum):
    """Supported model types for medical applications."""

    BERT = "bert"
    ROBERTA = "roberta"
    GPT2 = "gpt2"
    T5 = "t5"
    MEDICAL_BERT = "medical_bert"
    BIOBERT = "biobert"
    CLINICAL_BERT = "clinical_bert"
    PUBMED_BERT = "pubmed_bert"
    BLUEBERT = "bluebert"


class MedicalModelConfigSchema(BaseModel):
    """Pydantic schema for medical model configuration.

    This schema is used for validation and documentation of the
    medical model configuration.
    """

    # Model configuration
    model: str = Field(
        ...,
        description="Path to the model directory or model identifier.",
        examples=["bert-base-uncased"],
    )

    model_type: str = Field(
        default="base",
        description="Type of the model architecture.",
    )

    pretrained_model_name_or_path: Optional[str] = Field(
        default=None,
        description="Name or path of the pretrained model from Hugging Face Hub",
    )

    # Performance configuration
    max_num_batched_tokens: int = Field(
        default=32768,
        description="Maximum number of tokens to process in a single batch.",
        gt=0,
    )

    max_num_seqs: int = Field(
        default=512,
        description="Maximum number of sequences to process in parallel.",
        gt=0,
    )

    max_model_len: int = Field(
        default=4096, description="Maximum context length the model can handle.", gt=0
    )

    gpu_memory_utilization: float = Field(
        default=0.9, description="Fraction of GPU memory to use.", gt=0, le=1.0
    )

    tensor_parallel_size: int = Field(
        default=1,
        description="Number of GPUs to use for tensor parallelism.",
        ge=1,
        le=8,
    )

    # Medical-specific configuration
    max_medical_seq_length: int = Field(
        default=512,
        description="Maximum sequence length for medical text processing.",
        gt=0,
    )

    batch_size: int = Field(
        default=32, description="Default batch size for inference.", gt=0
    )

    enable_uncertainty_estimation: bool = Field(
        default=False,
        description="Whether to enable uncertainty estimation in model outputs",
    )

    uncertainty_threshold: float = Field(
        default=0.3,
        description="Threshold for model uncertainty calibration",
        ge=0.0,
        le=1.0,
    )

    cache_ttl: int = Field(
        default=3600, description="Time-to-live for cache in seconds", ge=0
    )

    # Medical domain configuration
    medical_specialties: List[str] = Field(
        default_factory=list,
        description="List of medical specialties this model is trained on",
    )

    anatomical_regions: List[str] = Field(
        default_factory=list,
        description="List of anatomical regions this model can process",
    )

    # NER configuration
    medical_entity_types: List[str] = Field(
        default_factory=list, description="Types of medical entities to recognize"
    )

    ner_confidence_threshold: float = Field(
        default=0.85,
        description="Minimum confidence score for NER predictions",
        ge=0.0,
        le=1.0,
    )

    max_entity_span_length: int = Field(
        default=10, description="Maximum token length for entity spans", gt=0
    )

    # Entity linking configuration
    entity_linking: EntityLinkingConfig = Field(
        default_factory=EntityLinkingConfig,
        description="Configuration for entity linking to knowledge bases",
    )

    # Document processing
    document_types: List[str] = Field(
        default_factory=list, description="Types of clinical documents supported"
    )

    section_headers: List[str] = Field(
        default_factory=list, description="Common section headers in clinical documents"
    )

    # API configuration
    max_retries: int = Field(
        default=3, description="Maximum number of retries for API calls", ge=0
    )

    request_timeout: int = Field(
        default=30, description="Timeout in seconds for API requests", ge=0
    )

    # Domain adaptation
    domain_adaptation: bool = Field(
        default=False, description="Whether to enable domain adaptation"
    )

    domain_adaptation_lambda: float = Field(
        default=0.1, description="Weight for domain adaptation loss", ge=0.0
    )

    domain_specific_vocab: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Domain-specific vocabulary terms"
    )

    # Domain configuration
    domain_config: DomainConfig = Field(
        default_factory=DomainConfig,
        description="Configuration for domain-specific settings",
    )

    # Compliance
    regulatory_compliance: List[str] = Field(
        default_factory=list, description="Regulatory standards the model complies with"
    )

    # Internal fields
    config_version: str = Field(
        default="1.0.0", description="Configuration schema version"
    )

    # Pydantic v2 style config
    model_config = ConfigDict(
        use_enum_values=True,
        extra="allow",  # Allow extra fields to support dynamic attributes
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "model": "medical-bert-base",
                "model_type": "bert",
                "max_medical_seq_length": 512,
                "batch_size": 32,
                "medical_specialties": ["cardiology", "radiology"],
                "anatomical_regions": ["chest", "abdomen"],
                "regulatory_compliance": ["hipaa", "gdpr"],
                "entity_linking": {
                    "enabled": True,
                    "knowledge_bases": ["UMLS", "SNOMED_CT"],
                    "confidence_threshold": 0.8,
                },
                "domain_config": {
                    "domain_adaptation": True,
                    "domain_adaptation_lambda": 0.1,
                    "domain_specific_vocab": None,
                },
            }
        },
    )

    def model_post_init(self, __context: Any) -> None:
        """Handle post-init configuration."""
        # Convert dict to EntityLinkingConfig if needed
        if isinstance(self.entity_linking, dict):
            self.entity_linking = EntityLinkingConfig(**self.entity_linking)

        # Convert dict to DomainConfig if needed
        if isinstance(self.domain_config, dict):
            self.domain_config = DomainConfig(**self.domain_config)

        # Ensure domain_specific_vocab is properly initialized
        if self.domain_specific_vocab is None:
            self.domain_specific_vocab = {}
