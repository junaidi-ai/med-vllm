"""
Data models for configuration types.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


@dataclass
class MetricRange:
    """Range for metric values with optional bounds."""

    min: float
    max: float
    unit: str


class MetricConfig(BaseModel):
    """Configuration for a single metric."""

    description: str
    unit: str
    normal_range: MetricRange
    critical_range: Optional[MetricRange] = None
    category: str
    higher_is_worse: bool = False
    required: bool = True


class ClinicalMetrics(BaseModel):
    """Configuration for clinical metrics."""

    vital_signs: Dict[str, MetricConfig]
    lab_tests: Dict[str, MetricConfig]
    scores: Dict[str, MetricConfig]


class DomainConfig(BaseModel):
    """Configuration for domain-specific settings."""

    domain_adaptation: bool = False
    domain_adaptation_lambda: float = 0.1
    domain_specific_vocab: Optional[Dict[str, List[str]]] = None
    # Allow model_* field names without protected namespace warnings
    model_config = ConfigDict(protected_namespaces=())


class EntityLinkingConfig(BaseModel):
    """Configuration for entity linking."""

    enabled: bool = False
    knowledge_bases: List[str] = []
    confidence_threshold: float = 0.8
    model_config = ConfigDict(protected_namespaces=())


class ModelConfig(BaseModel):
    """Base model configuration."""

    model_name: str = Field(..., description="Name or path of the model")
    model_type: str = Field(..., description="Type of the model architecture")
    max_sequence_length: int = Field(
        default=512, description="Maximum sequence length for the model"
    )
    do_lower_case: bool = Field(default=True, description="Whether to lowercase the input")
    use_fast_tokenizer: bool = Field(
        default=True, description="Whether to use fast tokenizer if available"
    )
    add_prefix_space: bool = Field(
        default=False, description="Whether to add a leading space to the first word"
    )
    # Suppress protected namespace warnings for fields like model_name/model_type
    model_config = ConfigDict(protected_namespaces=())
