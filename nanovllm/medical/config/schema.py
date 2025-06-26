"""Pydantic models for configuration validation."""
from typing import List, Optional, Dict, Any, ClassVar, Union, Annotated
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.functional_validators import AfterValidator
from pydantic.fields import FieldInfo
from pathlib import Path

class ModelType(str, Enum):
    """Supported model types."""
    BERT = "bert"
    CLINICAL_BERT = "clinical-bert"
    BIO_CLINICAL_BERT = "bio-clinical-bert"
    # Add more model types as needed

def normalize_string_list(value: str) -> List[str]:
    """Normalize a string or list of strings to a list of normalized strings."""
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [v.lower().replace(" ", "_") for v in value if v]

# Simple string types with validation
MedicalSpecialty = str
AnatomicalRegion = str

class MedicalModelConfigSchema(BaseModel):
    """Pydantic model for MedicalModelConfig validation."""
    # Pydantic v2 config
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='allow',  # Allow extra fields for backward compatibility
        json_schema_extra={
            "example": {
                "model_type": "bert",
                "model": "/path/to/model",
                "config_version": "0.1.0",
                "max_medical_seq_length": 512,
                "medical_specialties": ["cardiology", "neurology"],
                "anatomical_regions": ["head", "chest"],
                "enable_uncertainty_estimation": True,
                "use_crf": True,
                "do_lower_case": True,
                "preserve_case_for_abbreviations": True
            }
        }
    )
    
    # Core configuration
    model_type: ModelType = Field(
        default=ModelType.BERT,
        description="Type of the medical model"
    )
    model: str = Field(
        default="",
        description="Path to the model directory or model identifier"
    )
    config_version: str = Field(
        default="0.1.0",
        description="Configuration version"
    )
    
    # Model parameters
    max_medical_seq_length: int = Field(
        default=512,
        gt=0,
        le=4096,
        description="Maximum sequence length for medical text"
    )
    
    # Medical parameters
    medical_specialties: List[Union[MedicalSpecialty, str]] = Field(
        default_factory=list,
        description="List of medical specialties this model is trained on"
    )
    anatomical_regions: List[Union[AnatomicalRegion, str]] = Field(
        default_factory=list,
        description="List of anatomical regions this model is trained on"
    )
    enable_uncertainty_estimation: bool = Field(
        default=False,
        description="Whether to enable uncertainty estimation"
    )
    
    # Advanced parameters
    use_crf: bool = Field(
        default=True,
        description="Whether to use Conditional Random Field for sequence tagging"
    )
    do_lower_case: bool = Field(
        default=True,
        description="Whether to lowercase the input text"
    )
    preserve_case_for_abbreviations: bool = Field(
        default=True,
        description="Whether to preserve case for medical abbreviations"
    )
    
    # Validation methods
    @field_validator('model')
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Validate that the model path exists."""
        if not v:  # Skip validation if empty (will be set to cwd in post_init)
            return v
            
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model path does not exist: {v}")
        return str(path.absolute())
    
    @field_validator('config_version')
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate configuration version."""
        from .medical_config import ConfigVersionManager
        version_info = ConfigVersionManager.get_version_info(v)
        if version_info.status == "unsupported":
            raise ValueError(f"Unsupported configuration version: {v}")
        return v
    
    @field_validator('medical_specialties', mode='before')
    @classmethod
    def validate_medical_specialties(cls, v: Any) -> List[str]:
        """Normalize and validate medical specialties."""
        if v is None:
            return []
            
        # Only accept list or tuple of strings
        if not isinstance(v, (list, tuple)):
            raise ValueError("medical_specialties must be a list or tuple of strings")
            
        # Ensure all items are non-empty strings
        if not all(isinstance(s, str) and s.strip() for s in v):
            raise ValueError("All medical_specialties must be non-empty strings")
            
        # Normalize the values
        normalized = [s.lower().strip().replace(" ", "_") for s in v]
        if not normalized and v:  # This shouldn't happen due to the check above, but just in case
            raise ValueError("Invalid medical_specialties values")
            
        return normalized
    
    @field_validator('anatomical_regions', mode='before')
    @classmethod
    def validate_anatomical_regions(cls, v: Any) -> List[str]:
        """Normalize and validate anatomical regions."""
        if v is None:
            return []
            
        # Only accept list or tuple of strings
        if not isinstance(v, (list, tuple)):
            raise ValueError("anatomical_regions must be a list or tuple of strings")
            
        # Ensure all items are non-empty strings
        if not all(isinstance(r, str) and r.strip() for r in v):
            raise ValueError("All anatomical_regions must be non-empty strings")
            
        # Normalize the values
        normalized = [r.lower().strip().replace(" ", "_") for r in v]
        if not normalized and v:  # This shouldn't happen due to the check above, but just in case
            raise ValueError("Invalid anatomical_regions values")
            
        return normalized
