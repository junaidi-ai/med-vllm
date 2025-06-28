"""Pydantic models for configuration validation."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.fields import FieldInfo
from pydantic.functional_validators import AfterValidator


class ModelType(str, Enum):
    """Supported model types."""

    BERT = "bert"
    CLINICAL_BERT = "clinical-bert"
    BIO_CLINICAL_BERT = "bio-clinical-bert"
    # Add more model types as needed


def normalize_string_list(value: Union[str, List[str]]) -> List[str]:
    """Normalize a string or list of strings to a list of normalized strings.

    Args:
        value: Input string or list of strings to normalize

    Returns:
        List of normalized strings
    """
    if not value:
        return []
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).lower().replace(" ", "_") for v in value if v]


# Simple string types with validation
MedicalSpecialty = str
AnatomicalRegion = str


class MedicalModelConfigSchema(BaseModel):
    """Pydantic model for MedicalModelConfig validation."""

    # Pydantic v2 config
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra="allow",  # Allow extra fields for backward compatibility
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
                "preserve_case_for_abbreviations": True,
            }
        },
    )

    # Core configuration
    model_type: ModelType = Field(
        default=ModelType.BERT, description="Type of the medical model"
    )
    model: str = Field(
        default="", description="Path to the model directory or model identifier"
    )
    config_version: str = Field(default="0.1.0", description="Configuration version")

    # Model parameters
    max_medical_seq_length: int = Field(
        default=512,
        gt=0,
        le=4096,
        description="Maximum sequence length for medical text",
    )

    # Medical parameters
    medical_specialties: List[Union[MedicalSpecialty, str]] = Field(
        default_factory=list,
        description="List of medical specialties this model is trained on",
    )
    anatomical_regions: List[Union[AnatomicalRegion, str]] = Field(
        default_factory=list,
        description="List of anatomical regions this model is trained on",
    )
    enable_uncertainty_estimation: bool = Field(
        default=False, description="Whether to enable uncertainty estimation"
    )

    # Advanced parameters
    use_crf: bool = Field(
        default=True,
        description="Whether to use Conditional Random Field for sequence tagging",
    )
    do_lower_case: bool = Field(
        default=True, description="Whether to lowercase the input text"
    )
    preserve_case_for_abbreviations: bool = Field(
        default=True, description="Whether to preserve case for medical abbreviations"
    )

    # Validation methods
    @field_validator("model")
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Validate that the model path exists."""
        if not v:  # Skip validation if empty (will be set to cwd in post_init)
            return v

        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model path does not exist: {v}")
        return str(path.absolute())

    @field_validator("config_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate configuration version."""
        from .medical_config import ConfigVersionManager

        version_info = ConfigVersionManager.get_version_info(v)
        if version_info.status == "unsupported":
            raise ValueError(f"Unsupported configuration version: {v}")
        return v

    @field_validator("medical_specialties", mode="before")
    @classmethod
    def validate_medical_specialties(cls, v: Any) -> List[str]:
        """Normalize and validate medical specialties.

        Args:
            v: Input value which can be None, string, or list of strings

        Returns:
            List of normalized medical specialty strings

        Raises:
            ValueError: If the input is not a valid list of medical specialties
        """
        if v is None:
            return []

        # Handle string input (comma-separated values)
        if isinstance(v, str):
            v = [s.strip() for s in v.split(",") if s.strip()]

        # Only accept list or tuple of strings
        if not isinstance(v, (list, tuple)):
            raise ValueError(
                "medical_specialties must be a string, list, or tuple of strings"
            )

        # Convert all items to strings if they aren't already
        try:
            v = [str(item) for item in v]
        except (TypeError, ValueError) as e:
            raise ValueError(
                "All medical_specialties must be convertible to strings"
            ) from e

        # Ensure all items are non-empty after conversion
        if not all(s.strip() for s in v):
            raise ValueError("All medical_specialties must be non-empty strings")

        # Normalize the values (lowercase, replace spaces with underscores, etc.)
        normalized = []
        for s in v:
            normalized_s = s.strip().lower().replace(" ", "_")
            # Remove any non-alphanumeric characters except underscores
            normalized_s = "".join(c for c in normalized_s if c.isalnum() or c == "_")
            if normalized_s:  # Only add non-empty strings
                normalized.append(normalized_s)

        if not normalized and v:
            raise ValueError("No valid medical specialties found after normalization")

        # Remove duplicates while preserving order
        seen = set()
        return [x for x in normalized if not (x in seen or seen.add(x))]

    @field_validator("anatomical_regions", mode="before")
    @classmethod
    def validate_anatomical_regions(cls, v: Any) -> List[str]:
        """Normalize and validate anatomical regions.

        Args:
            v: Input value which can be None, string, or list of strings

        Returns:
            List of normalized anatomical region strings

        Raises:
            ValueError: If the input is not a valid list of anatomical regions
        """
        if v is None:
            return []

        # Handle string input (comma-separated values)
        if isinstance(v, str):
            v = [s.strip() for s in v.split(",") if s.strip()]

        # Only accept list or tuple of strings
        if not isinstance(v, (list, tuple)):
            raise ValueError(
                "anatomical_regions must be a string, list, or tuple of strings"
            )

        # Convert all items to strings if they aren't already
        try:
            v = [str(item) for item in v]
        except (TypeError, ValueError) as e:
            raise ValueError(
                "All anatomical_regions must be convertible to strings"
            ) from e

        # Ensure all items are non-empty after conversion
        if not all(s.strip() for s in v):
            raise ValueError("All anatomical_regions must be non-empty strings")

        # Normalize the values (lowercase, replace spaces with underscores, etc.)
        normalized = []
        for r in v:
            normalized_r = r.strip().lower().replace(" ", "_")
            # Remove any non-alphanumeric characters except underscores and hyphens
            normalized_r = "".join(c for c in normalized_r if c.isalnum() or c in "_-")
            if normalized_r:  # Only add non-empty strings
                normalized.append(normalized_r)

        if not normalized and v:
            raise ValueError("No valid anatomical regions found after normalization")

        # Remove duplicates while preserving order
        seen = set()
        return [x for x in normalized if not (x in seen or seen.add(x))]
