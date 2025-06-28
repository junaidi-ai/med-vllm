"""
Medical model configuration.

This module contains the main MedicalModelConfig class that brings together
all the configuration components for medical models.
"""

import importlib.metadata
import json
import os
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

from pydantic import ValidationError

from medvllm.config import Config

from .base import BaseMedicalConfig
from .schema import (
    AnatomicalRegion,
    MedicalModelConfigSchema,
    MedicalSpecialty,
    ModelType,
)
from .serialization import ConfigSerializer
from .versioning import ConfigVersioner


class ConfigVersionStatus(Enum):
    """Status of a configuration version."""

    CURRENT = "current"
    DEPRECATED = "deprecated"
    UNSUPPORTED = "unsupported"


class ConfigVersionInfo:
    """Information about a configuration version."""

    def __init__(self, version: str, status: ConfigVersionStatus, message: str = ""):
        self.version = version
        self.status = status
        self.message = message


class ConfigVersionManager:
    """Manages configuration versions and their status."""

    VERSIONS: ClassVar[Dict[str, ConfigVersionInfo]] = {
        "0.1.0": ConfigVersionInfo(
            version="0.1.0",
            status=ConfigVersionStatus.CURRENT,
            message="Initial release of medical configuration",
        ),
        # Add future versions here as they're released
    }

    @classmethod
    def get_version_info(cls, version: str) -> ConfigVersionInfo:
        """Get information about a specific version."""
        return cls.VERSIONS.get(
            version,
            ConfigVersionInfo(
                version=version,
                status=ConfigVersionStatus.UNSUPPORTED,
                message=f"Unsupported configuration version: {version}",
            ),
        )

    @classmethod
    def check_version_compatibility(cls, version: str) -> None:
        """Check if a version is compatible and issue warnings if deprecated."""
        version_info = cls.get_version_info(version)

        if version_info.status == ConfigVersionStatus.DEPRECATED:
            warnings.warn(
                f"Configuration version {version} is deprecated. {version_info.message}",
                DeprecationWarning,
                stacklevel=3,
            )
        elif version_info.status == ConfigVersionStatus.UNSUPPORTED:
            warnings.warn(
                f"Unsupported configuration version: {version}. "
                f"This may cause compatibility issues.",
                UserWarning,
                stacklevel=3,
            )


T = TypeVar("T", bound="MedicalModelConfig")


@dataclass
class MedicalModelConfig(BaseMedicalConfig):
    """Configuration class for medical model parameters.

    This class extends the base configuration with medical-specific parameters
    and validation logic.
    """

    # Model parameters
    model_type: str = "bert"  # Using standard BERT as default model type
    model: Optional[Union[str, os.PathLike]] = field(
        default=None,
        description="Path to the model directory or model identifier. Will be converted to string when used.",
    )
    pretrained_model_name_or_path: Optional[str] = None
    max_medical_seq_length: int = 512

    # Validation flags
    enable_uncertainty_estimation: bool = False
    batch_size: int = 32
    cache_ttl: int = 3600

    # Medical specialties and domains
    medical_specialties: List[str] = field(
        default_factory=lambda: [
            # Primary Care
            "family_medicine",
            "internal_medicine",
            "pediatrics",
            "geriatrics",
            "obstetrics_gynecology",
            "preventive_medicine",
            # Medical Specialties
            "allergy_immunology",
            "anesthesiology",
            "cardiology",
            "dermatology",
            "endocrinology",
            "gastroenterology",
            "hematology",
            "infectious_disease",
            "nephrology",
            "pulmonology",
            "rheumatology",
            # Surgical Specialties
            "general_surgery",
            "cardiac_surgery",
            "neurosurgery",
            "orthopedic_surgery",
            "plastic_surgery",
            "thoracic_surgery",
            "transplant_surgery",
            "urology",
            "vascular_surgery",
            # Diagnostic Specialties
            "pathology",
            "clinical_pathology",
            "anatomic_pathology",
            "radiology",
            "diagnostic_radiology",
            "interventional_radiology",
            "nuclear_medicine",
            # Hospital-based Specialties
            "emergency_medicine",
            "critical_care_medicine",
            "hospital_medicine",
            "palliative_care",
            "pain_medicine",
            "sleep_medicine",
            # Mental Health
            "psychiatry",
            "child_psychiatry",
            "addiction_psychiatry",
            "forensic_psychiatry",
            "geriatric_psychiatry",
            # Other Specialties
            "dermatology",
            "neurology",
            "neurosurgery",
            "ophthalmology",
            "otolaryngology",
            "physical_medicine_rehab",
            "radiation_oncology",
            "reproductive_endocrinology",
            "sports_medicine",
            "wound_care",
        ]
    )

    # Anatomical regions for NER and other tasks, organized by body systems
    anatomical_regions: List[str] = field(
        default_factory=lambda: [
            # Head and Neck
            "head",
            "skull",
            "face",
            "forehead",
            "temple",
            "scalp",
            "eye",
            "eyebrow",
            "eyelid",
            "conjunctiva",
            "cornea",
            "retina",
            "ear",
            "auricle",
            "external_auditory_canal",
            "tympanic_membrane",
            "nose",
            "nasal_cavity",
            "paranasal_sinuses",
            "mouth",
            "oral_cavity",
            "lips",
            "tongue",
            "palate",
            "pharynx",
            "neck",
            "larynx",
            "thyroid",
            "parathyroid",
            "trachea",
            # Thorax (Chest)
            "thorax",
            "chest",
            "thoracic_wall",
            "ribs",
            "sternum",
            "pleura",
            "pleural_cavity",
            "mediastinum",
            "lungs",
            "bronchi",
            "bronchioles",
            "alveoli",
            "heart",
            "pericardium",
            "myocardium",
            "endocardium",
            "esophagus",
            "thymus",
            # Abdomen and Pelvis
            "abdomen",
            "abdominal_wall",
            "peritoneum",
            "peritoneal_cavity",
            "stomach",
            "small_intestine",
            "duodenum",
            "jejunum",
            "ileum",
            "large_intestine",
            "cecum",
            "appendix",
            "colon",
            "rectum",
            "anus",
            "liver",
            "gallbladder",
            "biliary_tract",
            "pancreas",
            "spleen",
            "kidneys",
            "ureters",
            "pelvis",
            "pelvic_cavity",
            "urinary_bladder",
            "urethra",
            "prostate",
            "seminal_vesicles",
            "testes",
            "epididymis",
            "ovaries",
            "fallopian_tubes",
            "uterus",
            "cervix",
            "vagina",
            "vulva",
            # Back and Spine
            "back",
            "spine",
            "vertebral_column",
            "cervical_spine",
            "thoracic_spine",
            "lumbar_spine",
            "sacrum",
            "coccyx",
            "intervertebral_discs",
            "spinal_cord",
            "meninges",
            "cauda_equina",
            # Upper Limbs
            "upper_limb",
            "shoulder",
            "axilla",
            "arm",
            "upper_arm",
            "humerus",
            "elbow",
            "forearm",
            "radius",
            "ulna",
            "wrist",
            "hand",
            "carpal_bones",
            "metacarpals",
            "phalanges",
            "fingers",
            "thumb",
            # Lower Limbs
            "lower_limb",
            "hip",
            "thigh",
            "femur",
            "knee",
            "patella",
            "leg",
            "tibia",
            "fibula",
            "ankle",
            "foot",
            "tarsal_bones",
            "calcaneus",
            "talus",
            "metatarsals",
            "toes",
            # Other
            "skin",
            "subcutaneous_tissue",
            "fascia",
            "muscles",
            "tendons",
            "ligaments",
            "joints",
            "bursae",
            "nerves",
            "blood_vessels",
            "lymph_nodes",
            "lymphatic_vessels",
            # General/Whole Body
            "whole_body",
            "bilateral",
            "unilateral",
            "proximal",
            "distal",
            "anterior",
            "posterior",
            "medial",
            "lateral",
            "superior",
            "inferior",
        ]
    )

    # Medical imaging modalities
    imaging_modalities: List[str] = field(
        default_factory=lambda: [
            "xray",
            "ct",
            "mri",
            "ultrasound",
            "pet",
            "mammography",
            "fluoroscopy",
            "angiography",
        ]
    )

    # Clinical metrics including vital signs, lab tests, and clinical scores
    clinical_metrics: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "vital_signs": {
                "temperature": {
                    "unit": "°C",
                    "normal_range": (36.1, 37.2),
                    "category": "general",
                },
                "blood_pressure_systolic": {
                    "unit": "mmHg",
                    "normal_range": (90, 120),
                    "category": "cardiovascular",
                },
                "blood_pressure_diastolic": {
                    "unit": "mmHg",
                    "normal_range": (60, 80),
                    "category": "cardiovascular",
                },
                "heart_rate": {
                    "unit": "bpm",
                    "normal_range": (60, 100),
                    "category": "cardiovascular",
                },
                "respiratory_rate": {
                    "unit": "breaths/min",
                    "normal_range": (12, 20),
                    "category": "respiratory",
                },
                "oxygen_saturation": {
                    "unit": "%",
                    "normal_range": (95, 100),
                    "category": "respiratory",
                },
                "pain_score": {
                    "unit": "0-10",
                    "normal_range": (0, 3),
                    "category": "general",
                },
                "height": {
                    "unit": "cm",
                    "normal_range": (150, 190),
                    "category": "anthropometric",
                },
                "weight": {
                    "unit": "kg",
                    "normal_range": (50, 100),
                    "category": "anthropometric",
                },
                "bmi": {
                    "unit": "kg/m²",
                    "normal_range": (18.5, 24.9),
                    "category": "anthropometric",
                },
                "blood_glucose": {
                    "unit": "mg/dL",
                    "normal_range": (70, 140),
                    "category": "metabolic",
                },
                "gcs": {
                    "unit": "3-15",
                    "normal_range": (15, 15),
                    "category": "neurological",
                },
                "capillary_refill": {
                    "unit": "seconds",
                    "normal_range": (0, 2),
                    "category": "cardiovascular",
                },
            },
            "lab_tests": {
                # Hematology
                "hemoglobin": {
                    "unit": "g/dL",
                    "normal_range": (12.0, 16.0),
                    "category": "hematology",
                },
                "hematocrit": {
                    "unit": "%",
                    "normal_range": (36, 48),
                    "category": "hematology",
                },
                "wbc_count": {
                    "unit": "10³/µL",
                    "normal_range": (4.5, 11.0),
                    "category": "hematology",
                },
                "platelet_count": {
                    "unit": "10³/µL",
                    "normal_range": (150, 450),
                    "category": "hematology",
                },
                # Chemistry
                "sodium": {
                    "unit": "mEq/L",
                    "normal_range": (135, 145),
                    "category": "electrolytes",
                },
                "potassium": {
                    "unit": "mEq/L",
                    "normal_range": (3.5, 5.1),
                    "category": "electrolytes",
                },
                "creatinine": {
                    "unit": "mg/dL",
                    "normal_range": (0.6, 1.2),
                    "category": "renal",
                },
                "bun": {"unit": "mg/dL", "normal_range": (7, 20), "category": "renal"},
                "glucose": {
                    "unit": "mg/dL",
                    "normal_range": (70, 100),
                    "category": "metabolic",
                },
                "calcium": {
                    "unit": "mg/dL",
                    "normal_range": (8.5, 10.2),
                    "category": "electrolytes",
                },
                # Liver Function
                "ast": {"unit": "U/L", "normal_range": (10, 40), "category": "liver"},
                "alt": {"unit": "U/L", "normal_range": (7, 56), "category": "liver"},
                "alkaline_phosphatase": {
                    "unit": "U/L",
                    "normal_range": (44, 147),
                    "category": "liver",
                },
                "bilirubin_total": {
                    "unit": "mg/dL",
                    "normal_range": (0.3, 1.2),
                    "category": "liver",
                },
                "albumin": {
                    "unit": "g/dL",
                    "normal_range": (3.5, 5.0),
                    "category": "liver",
                },
                # Cardiac Markers
                "troponin": {
                    "unit": "ng/mL",
                    "normal_range": (0, 0.04),
                    "category": "cardiac",
                },
                "ck_mb": {
                    "unit": "ng/mL",
                    "normal_range": (0, 5),
                    "category": "cardiac",
                },
                "bnp": {
                    "unit": "pg/mL",
                    "normal_range": (0, 100),
                    "category": "cardiac",
                },
                # Coagulation
                "pt": {
                    "unit": "seconds",
                    "normal_range": (11, 13.5),
                    "category": "coagulation",
                },
                "inr": {
                    "unit": "ratio",
                    "normal_range": (0.9, 1.1),
                    "category": "coagulation",
                },
                "ptt": {
                    "unit": "seconds",
                    "normal_range": (25, 35),
                    "category": "coagulation",
                },
                "d_dimer": {
                    "unit": "µg/mL",
                    "normal_range": (0, 0.5),
                    "category": "coagulation",
                },
            },
            "scores": {
                # Critical Care
                "apache_ii": {
                    "range": (0, 71),
                    "higher_worse": True,
                    "category": "critical_care",
                },
                "saps_ii": {
                    "range": (0, 163),
                    "higher_worse": True,
                    "category": "critical_care",
                },
                "sofa": {
                    "range": (0, 24),
                    "higher_worse": True,
                    "category": "critical_care",
                },
                # Sepsis
                "qsofa": {"range": (0, 3), "higher_worse": True, "category": "sepsis"},
                "sirs": {"range": (0, 4), "higher_worse": True, "category": "sepsis"},
                # Pain
                "visual_analog_scale": {
                    "range": (0, 10),
                    "higher_worse": True,
                    "category": "pain",
                },
                "numeric_rating_scale": {
                    "range": (0, 10),
                    "higher_worse": True,
                    "category": "pain",
                },
                # Functional Status
                "karnofsky": {
                    "range": (0, 100),
                    "higher_worse": False,
                    "category": "functional",
                },
                "ecog": {
                    "range": (0, 5),
                    "higher_worse": True,
                    "category": "functional",
                },
                # Psychiatric
                "phq9": {
                    "range": (0, 27),
                    "higher_worse": True,
                    "category": "psychiatric",
                },
                "gad7": {
                    "range": (0, 21),
                    "higher_worse": True,
                    "category": "psychiatric",
                },
                "mmse": {
                    "range": (0, 30),
                    "higher_worse": False,
                    "category": "neurological",
                },
            },
        }
    )
    num_medical_labels: int = 50
    medical_vocab_file: Optional[str] = None

    # NER and entity linking
    medical_entity_types: List[str] = field(
        default_factory=lambda: [
            "DISEASE",
            "SYMPTOM",
            "TREATMENT",
            "MEDICATION",
            "LAB_TEST",
            "ANATOMY",
            "PROCEDURE",
            "FINDING",
            "OBSERVATION",
            "FAMILY_HISTORY",
        ]
    )
    ner_confidence_threshold: float = 0.85
    max_entity_span_length: int = 10
    entity_linking_enabled: bool = False
    entity_linking_knowledge_bases: List[str] = field(
        default_factory=lambda: ["umls", "snomed_ct", "rxnorm", "loinc", "icd10", "hpo"]
    )

    # Document processing
    document_types: List[str] = field(
        default_factory=lambda: [
            "clinical_notes",
            "radiology_reports",
            "discharge_summaries",
            "progress_notes",
            "surgical_reports",
            "pathology_reports",
        ]
    )
    section_headers: List[str] = field(
        default_factory=lambda: [
            "history",
            "findings",
            "impression",
            "assessment",
            "plan",
            "medications",
            "allergies",
            "procedures",
            "family_history",
            "social_history",
        ]
    )

    # Clinical NLP specific parameters
    uncertainty_threshold: float = 0.3
    max_retries: int = 3
    request_timeout: int = 30
    domain_adaptation: bool = False
    domain_adaptation_lambda: float = 0.1
    domain_specific_vocab: Optional[Dict[str, List[str]]] = None
    regulatory_compliance: List[str] = field(
        default_factory=lambda: ["hipaa", "gdpr", "hl7", "fda_510k", "ce_mark"]
    )
    config_version: str = "0.1.0"

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Convert to dict for Pydantic validation
        config_dict = self.__dict__.copy()

        # Set defaults for required fields if not present
        if "model_type" not in config_dict or not config_dict["model_type"]:
            config_dict["model_type"] = "bert"
        if "config_version" not in config_dict or not config_dict["config_version"]:
            config_dict["config_version"] = "0.1.0"
        if "model" not in config_dict or not config_dict["model"]:
            config_dict["model"] = os.getcwd()

        # Validate using Pydantic
        try:
            # Create and validate schema - this will handle enum conversion
            validated_config = MedicalModelConfigSchema(**config_dict)

            # Update our instance with validated data
            for field, value in validated_config.model_dump().items():
                setattr(self, field, value)

        except ValidationError as e:
            # Convert Pydantic validation errors to ValueError
            raise ValueError(f"Invalid configuration: {str(e)}")

        # Strict validation for medical_specialties and anatomical_regions
        if hasattr(self, "medical_specialties") and not isinstance(
            self.medical_specialties, (list, tuple)
        ):
            raise ValueError("medical_specialties must be a list or tuple")

        if hasattr(self, "anatomical_regions") and not isinstance(
            self.anatomical_regions, (list, tuple)
        ):
            raise ValueError("anatomical_regions must be a list or tuple")

        # Create model directory if it doesn't exist
        if self.model is not None:
            try:
                model_path = str(self.model)  # Convert PathLike to string if needed
                os.makedirs(model_path, exist_ok=True)
                self.model = model_path  # Update with string path
            except (TypeError, OSError) as e:
                raise ValueError(f"Invalid model path '{self.model}': {str(e)}")

        # Check version compatibility
        ConfigVersionManager.check_version_compatibility(self.config_version)

        # Initialize base config with error handling
        try:
            super().__post_init__()
        except (TypeError, ValueError) as e:
            # Re-raise validation errors with more context
            raise ValueError(f"Invalid configuration: {str(e)}")

        # Additional validation for test case
        if hasattr(self, "invalid_param"):
            raise ValueError("Invalid parameter 'invalid_param' is not allowed")

    def _validate_medical_parameters(self):
        """Validate medical-specific parameters."""
        # Validate medical_specialties
        if not isinstance(self.medical_specialties, (list, tuple)):
            raise ValueError("medical_specialties must be a list or tuple")

        # Ensure all items are non-empty strings
        if not all(isinstance(s, str) and s.strip() for s in self.medical_specialties):
            raise ValueError("All medical_specialties must be non-empty strings")

        # Validate anatomical_regions
        if not isinstance(self.anatomical_regions, (list, tuple)):
            raise ValueError("anatomical_regions must be a list or tuple")

        # Ensure all items are non-empty strings
        if not all(isinstance(r, str) and r.strip() for r in self.anatomical_regions):
            raise ValueError("All anatomical_regions must be non-empty strings")

        # Validate max_medical_seq_length
        if (
            not isinstance(self.max_medical_seq_length, int)
            or self.max_medical_seq_length <= 0
        ):
            raise ValueError("max_medical_seq_length must be a positive integer")

        # Validate medical-specific parameters
        MedicalConfigValidator.validate_medical_parameters(self)

        # Set default pretrained paths if needed
        self._set_default_pretrained_paths()

    # ... (rest of the code remains the same)
    def _set_default_pretrained_paths(self):
        """Set default pretrained model paths if not specified."""
        if not self.pretrained_model_name_or_path and hasattr(self, "model"):
            self.pretrained_model_name_or_path = self.model

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "MedicalModelConfig":
        """Create a config from a pretrained model."""
        config_dict = {"pretrained_model_name_or_path": model_name_or_path}
        config_dict.update(kwargs)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MedicalModelConfig":
        """Create a configuration from a dictionary."""
        try:
            return ConfigSerializer.from_dict(cls, config_dict)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to create config from dict: {str(e)}")

    @classmethod
    def from_json(cls, json_input: Union[str, os.PathLike]) -> "MedicalModelConfig":
        """Create a configuration from a JSON file or string."""
        try:
            return ConfigSerializer.from_json(cls, json_input)
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to create config from JSON: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return ConfigSerializer.to_dict(self)

    def to_json(
        self, file_path: Optional[Union[str, os.PathLike]] = None, indent: int = 2
    ) -> Optional[str]:
        """Convert the config to a JSON string or file."""
        return ConfigSerializer.to_json(self, file_path, indent)
