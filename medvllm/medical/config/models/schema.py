"""
Schema definitions for medical model configuration.

This module contains Pydantic models and schemas for validating
and documenting the medical model configuration.
"""

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


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
    model: str = Field(
        ...,
        description="Path to the model directory or model identifier.",
        example="bert-base-uncased",
    )
    
    model_type: ModelType = Field(
        default=ModelType.BERT,
        description="Type of the model architecture.",
    )
    
    max_sequence_length: int = Field(
        default=512,
        description="Maximum sequence length for the model.",
        ge=32,
        le=4096,
    )
    
    batch_size: int = Field(
        default=32,
        description="Batch size for training and evaluation.",
        gt=0,
    )
    
    learning_rate: float = Field(
        default=5e-5,
        description="Learning rate for the optimizer.",
        gt=0.0,
    )
    
    num_train_epochs: int = Field(
        default=3,
        description="Number of training epochs.",
        ge=1,
    )
    
    class Config:
        """Pydantic config class."""
        use_enum_values = True
        extra = "forbid"  # Prevent extra fields
        
        @classmethod
        def schema_extra(cls, schema: Dict, model: type) -> None:
            """Add example to the schema."""
            schema["example"] = {
                "model": "bert-base-uncased",
                "model_type": "bert",
                "max_sequence_length": 512,
                "batch_size": 32,
                "learning_rate": 5e-5,
                "num_train_epochs": 3,
            }
