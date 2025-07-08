"""
Medical model configuration.

This module contains the main MedicalModelConfig class that brings together
all the configuration components for medical models with enhanced type safety
and organization using the new modules.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AnyStr,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    final,
    get_args,
    get_origin,
    get_type_hints,
    overload,
    runtime_checkable,
)

# Type variable for generic types
T = TypeVar("T")

# Import yaml with type checking
try:
    import yaml
    from yaml import CSafeDumper as SafeDumper
    from yaml import CSafeLoader as SafeLoader
    from yaml import dump as yaml_dump
    from yaml import load as yaml_load
    from yaml import safe_dump as yaml_safe_dump
    from yaml import safe_load as yaml_safe_load
    from yaml.constructor import ConstructorError as YAMLConstructorError
    from yaml.error import MarkedYAMLError, YAMLError
    from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
    from yaml.representer import RepresenterError as YAMLRepresenterError
    from yaml.resolver import Resolver as YAMLResolver
    from yaml.scanner import ScannerError as YAMLScannerError
    from yaml.serializer import Serializer as YAMLSerializer
    from yaml.tokens import Token as YAMLToken
    from yaml.tokens import TokenType as YAMLTokenType
    from yaml.tokens import *  # noqa: F403

    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore
    YAML_AVAILABLE = False

from medvllm.medical.config.base import BaseMedicalConfig

# Local imports
from medvllm.medical.config.types import (
    AnatomicalRegion,
    DocumentType,
    EntityType,
    ImagingModality,
    MedicalSpecialty,
    RegulatoryStandard,
)
from medvllm.utils.logging import get_logger

# Constants for default values
DEFAULT_MODEL_TYPE = "medical_llm"
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_BATCH_SIZE = 1
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_UNCERTAINTY_THRESHOLD = 0.7
DEFAULT_NER_THRESHOLD = 0.5
DEFAULT_MAX_ENTITY_SPAN_LENGTH = 10

# Define supported model types as a set of strings
SUPPORTED_MODEL_TYPES = {
    "medical_llm",
    "medical_ner",
    "medical_qa",
    "medical_summarization",
    "medical_classification",
    "radiology_report",
    "clinical_notes",
}

# Default configuration values
DEFAULT_MEDICAL_SPECIALTIES = ["general_medicine"]
DEFAULT_ANATOMICAL_REGIONS = ["full_body"]
DEFAULT_IMAGING_MODALITIES = ["xray", "ct", "mri", "ultrasound"]
DEFAULT_ENTITY_TYPES = ["disease", "symptom", "medication", "procedure"]
DEFAULT_DOCUMENT_TYPES = ["clinical_note", "discharge_summary", "radiology_report"]
DEFAULT_SECTION_HEADERS = [
    "history_of_present_illness",
    "past_medical_history",
    "medications",
    "allergies",
    "family_history",
    "social_history",
    "review_of_systems",
    "physical_exam",
    "assessment_and_plan",
]

# Configuration version
CONFIG_VERSION = "1.0.0"

# Initialize logger
logger = get_logger(__name__)

# Type variable for class methods that return an instance of the class
T = TypeVar("T", bound="MedicalModelConfig")


@dataclass
class MedicalModelConfig(BaseMedicalConfig):
    """Configuration class for medical model parameters.

    This class extends the base configuration with medical-specific parameters
    and validation logic, utilizing the new modular structure for better
    organization and type safety.

    Attributes:
        model_type (str): Type of the model architecture. Must be one of the
            supported model types.
        model (str): Path to the model directory or model identifier.
        pretrained_model_name_or_path (Optional[str]): Name or path of the
            pretrained model.
        max_medical_seq_length (int): Maximum sequence length for medical text
            processing.
        batch_size (int): Default batch size for inference.
        enable_uncertainty_estimation (bool): Whether to enable uncertainty
            estimation.
        uncertainty_threshold (float): Threshold for model uncertainty
            calibration.
        cache_ttl (int): Time-to-live for cache in seconds.
        medical_specialties (List[Union[MedicalSpecialty, str]]): List of
            medical specialties.
        anatomical_regions (List[Union[AnatomicalRegion, str]]): List of
            anatomical regions.
        imaging_modalities (List[Union[ImagingModality, str]]): List of
            imaging modalities.
        medical_entity_types (List[Union[EntityType, str]]): Types of medical
            entities to recognize.
        ner_confidence_threshold (float): Minimum confidence score for NER
            predictions.
        max_entity_span_length (int): Maximum token length for entity spans.
        entity_linking (Dict[str, Any]): Configuration for entity linking.
        document_types (List[Union[DocumentType, str]]): Types of clinical
            documents supported.
        section_headers (List[str]): Common section headers in clinical
            documents.
        max_retries (int): Maximum number of retries for API calls.
        request_timeout (int): Timeout in seconds for API requests.
        domain_adaptation (bool): Whether to enable domain adaptation.
        domain_adaptation_lambda (float): Weight for domain adaptation loss.
        domain_specific_vocab (Optional[Dict[str, List[str]]]): Domain-specific
            vocabulary terms.
        regulatory_compliance (List[Union[RegulatoryStandard, str]]):
            Regulatory standards the model complies with.
        config_version (str): Configuration schema version.
    """

    # Model configuration
    model: str = field(
        default="",
        metadata={
            "description": (
                "Path to the model directory or model identifier. " "Required field."
            ),
            "required": True,
        },
    )

    model_type: str = field(
        default=DEFAULT_MODEL_TYPE,
        metadata={
            "description": (
                f"Type of the model architecture. Must be one of: "
                f"{', '.join(SUPPORTED_MODEL_TYPES)}"
            ),
            "choices": SUPPORTED_MODEL_TYPES,
        },
    )

    pretrained_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "description": (
                "Name or path of the pretrained model from Hugging Face Hub"
            )
        },
    )

    # Model architecture parameters
    max_medical_seq_length: int = field(
        default=DEFAULT_MAX_SEQ_LENGTH,
        metadata={
            "description": "Maximum sequence length for medical text " "processing"
        },
    )

    # Training and inference parameters
    batch_size: int = field(
        default=DEFAULT_BATCH_SIZE,
        metadata={"description": "Default batch size for inference"},
    )

    enable_uncertainty_estimation: bool = field(
        default=False,
        metadata={
            "description": ("Whether to enable uncertainty estimation in model outputs")
        },
    )

    uncertainty_threshold: float = field(
        default=DEFAULT_UNCERTAINTY_THRESHOLD,
        metadata={"description": "Threshold for model uncertainty calibration"},
    )

    # Cache settings
    cache_ttl: int = field(
        default=3600,
        metadata={"description": "Time-to-live for cache in seconds"},
    )

    # Medical domain configuration
    medical_specialties: List[Union[MedicalSpecialty, str]] = field(
        default_factory=lambda: list(DEFAULT_MEDICAL_SPECIALTIES),
        metadata={
            "description": "Medical specialties this model is trained on",
            "category": "medical_domain",
        },
    )

    anatomical_regions: List[Union[AnatomicalRegion, str]] = field(
        default_factory=lambda: list(DEFAULT_ANATOMICAL_REGIONS),
        metadata={
            "description": "List of anatomical regions this model can process",
            "category": "anatomy",
        },
    )

    imaging_modalities: List[Union[ImagingModality, str]] = field(
        default_factory=lambda: list(DEFAULT_IMAGING_MODALITIES),
        metadata={
            "description": "Medical imaging modalities supported by the model",
            "category": "imaging",
        },
    )

    # Clinical and entity recognition
    medical_entity_types: List[Union[EntityType, str]] = field(
        default_factory=lambda: list(DEFAULT_ENTITY_TYPES),
        metadata={"description": "Types of medical entities to recognize"},
    )

    ner_confidence_threshold: float = field(
        default=DEFAULT_NER_THRESHOLD,
        metadata={"description": "Minimum confidence score for NER predictions"},
    )

    max_entity_span_length: int = field(
        default=DEFAULT_MAX_ENTITY_SPAN_LENGTH,
        metadata={"description": "Maximum token length for entity spans"},
    )

    entity_linking: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "knowledge_bases": ["umls", "snomed_ct", "loinc"],
            "confidence_threshold": 0.8,
        },
        metadata={
            "description": "Configuration for entity linking to " "knowledge bases"
        },
    )

    # Document processing
    document_types: List[Union[DocumentType, str]] = field(
        default_factory=lambda: list(DEFAULT_DOCUMENT_TYPES),
        metadata={"description": "Types of clinical documents supported"},
    )

    section_headers: List[str] = field(
        default_factory=lambda: list(DEFAULT_SECTION_HEADERS),
        metadata={"description": "Common section headers in clinical documents"},
    )

    # API and request handling
    max_retries: int = field(
        default=3,  # Default max retries
        metadata={"description": "Maximum number of retries for API calls"},
    )

    request_timeout: int = field(
        default=30,  # Default request timeout in seconds
        metadata={"description": "Timeout in seconds for API requests"},
    )

    # Domain adaptation
    domain_adaptation: bool = field(
        default=False, metadata={"description": "Whether to enable domain adaptation"}
    )

    domain_adaptation_lambda: float = field(
        default=0.1, metadata={"description": "Weight for domain adaptation loss"}
    )

    domain_specific_vocab: Optional[Dict[str, List[str]]] = field(
        default=None, metadata={"description": "Domain-specific vocabulary terms"}
    )

    # Compliance and versioning
    regulatory_compliance: List[Union[RegulatoryStandard, str]] = field(
        default_factory=lambda: ["hipaa", "gdpr"],
        metadata={"description": "Regulatory standards the model complies with"},
    )

    config_version: str = field(
        default=CONFIG_VERSION,
        metadata={
            "description": "Configuration schema version",
            "readonly": True,
        },
    )

    def __post_init__(self) -> None:
        """Post-initialization validation and setup.

        This method performs several important tasks after the configuration is
        initialized:
        1. Validates all configuration parameters
        2. Sets up default values for optional fields
        3. Initializes version compatibility checks
        4. Performs any necessary type conversions
        """
        # Convert string values to proper enum types
        self._convert_enums()

        # Skip version compatibility check since we're removing the dependency
        # Just ensure config_version is set
        if not hasattr(self, "config_version"):
            self.config_version = "1.0.0"

        # Set default pretrained paths if not specified
        self._set_default_pretrained_paths()

        # Initialize any dependent configs
        self._initialize_dependent_configs()

        # Validate all parameters
        self._validate_medical_parameters()

    def _convert_enums(self) -> None:
        """Convert string values to proper enum types if needed.

        This ensures type safety and consistency when loading serialized data.
        """

        # Convert string values in lists to enums with type checking
        def safe_convert(
            value: Union[str, Enum], enum_type: Type[Enum]
        ) -> Union[Enum, str]:
            if isinstance(value, str):
                try:
                    return enum_type(value.upper())
                except (ValueError, AttributeError):
                    warnings.warn(
                        f"Invalid {enum_type.__name__} value: {value}",
                        UserWarning,
                        stacklevel=2,
                    )
            return value

        # Convert each list of enums to their string representations
        if self.medical_specialties:
            self.medical_specialties = [
                (
                    str(safe_convert(spec, MedicalSpecialty))
                    if isinstance(spec, str)
                    else str(spec.value) if hasattr(spec, "value") else str(spec)
                )
                for spec in self.medical_specialties
            ]

        if self.anatomical_regions:
            self.anatomical_regions = [
                (
                    str(safe_convert(region, AnatomicalRegion))
                    if isinstance(region, str)
                    else str(region.value) if hasattr(region, "value") else str(region)
                )
                for region in self.anatomical_regions
            ]

        if self.imaging_modalities:
            self.imaging_modalities = [
                (
                    str(safe_convert(modality, ImagingModality))
                    if isinstance(modality, str)
                    else (
                        str(modality.value)
                        if hasattr(modality, "value")
                        else str(modality)
                    )
                )
                for modality in self.imaging_modalities
            ]

        if self.medical_entity_types:
            self.medical_entity_types = [
                (
                    str(safe_convert(entity_type, EntityType))
                    if isinstance(entity_type, str)
                    else (
                        str(entity_type.value)
                        if hasattr(entity_type, "value")
                        else str(entity_type)
                    )
                )
                for entity_type in self.medical_entity_types
            ]

        if self.document_types:
            self.document_types = [
                (
                    str(safe_convert(doc_type, DocumentType))
                    if isinstance(doc_type, str)
                    else (
                        str(doc_type.value)
                        if hasattr(doc_type, "value")
                        else str(doc_type)
                    )
                )
                for doc_type in self.document_types
            ]

        if self.regulatory_compliance:
            self.regulatory_compliance = [
                (
                    str(safe_convert(std, RegulatoryStandard))
                    if isinstance(std, str)
                    else str(std.value) if hasattr(std, "value") else str(std)
                )
                for std in self.regulatory_compliance
            ]

    def copy(self) -> "MedicalModelConfig":
        """Create a copy of the configuration.

        This method ensures that domain_config and other fields are properly
        handled during copying.

        Returns:
            A new instance with the same parameters
        """
        # Get the dictionary representation
        data = self.to_dict()

        # Create a new instance using from_dict which handles all the
        # special cases
        return self.__class__.from_dict(data)

    def _initialize_dependent_configs(self) -> None:
        """Initialize any dependent configuration objects.

        This method sets up complex nested configurations that depend on other
        configuration values.
        """
        # Initialize domain config if needed
        if not hasattr(self, "domain_config") or self.domain_config is None:
            self.domain_config: Dict[str, Any] = {
                "enabled": self.domain_adaptation,
                "lambda_val": self.domain_adaptation_lambda,
                "vocab": self.domain_specific_vocab or {},
            }

        # Handle entity linking configuration
        if isinstance(self.entity_linking, dict):
            self.entity_linking = dict(self.entity_linking)

        # Skip Pydantic validation since we're removing the dependency
        # Just ensure required fields are present
        required_fields = ["model_type", "model"]
        for field_name in required_fields:
            if field_name not in self.__dict__:
                raise ValueError(f"Missing required field: {field_name}")

        # Create model directory if it doesn't exist
        if self.model is not None:
            try:
                # Convert PathLike to string if needed
                model_path = str(self.model)
                os.makedirs(model_path, exist_ok=True)
                self.model = model_path  # Update with string path
            except (TypeError, OSError) as e:
                raise ValueError(f"Invalid model path '{self.model}': {str(e)}")

        # Initialize base config with error handling
        try:
            super().__post_init__()
        except (TypeError, ValueError) as e:
            # Re-raise validation errors with more context
            raise ValueError(f"Invalid configuration: {str(e)}")

        # Additional validation for test case
        if hasattr(self, "invalid_param"):
            raise ValueError("Invalid parameter 'invalid_param' is not allowed")

    def _validate_medical_parameters(self) -> None:
        """Validate medical-specific parameters.

        This method validates all medical-specific parameters to ensure they
        have valid values. It raises ValueError for invalid values and issues
        warnings for potentially problematic but technically valid values.

        The validation includes:
        - Model type and architecture parameters
        - Medical domain-specific parameters
        - Clinical entity recognition settings
        - Compliance and regulatory settings
        - Performance and resource constraints

        Raises:
            ValueError: If any parameter has an invalid value
            UserWarning: For potentially problematic but technically valid
                values
        """
        # Base model validation
        self._validate_model_parameters()

        # Medical domain validation
        self._validate_medical_domain_parameters()

        # Clinical entity recognition validation
        self._validate_ner_parameters()

        # Performance and resource validation
        self._validate_performance_parameters()

        # Compliance and regulatory validation
        self._validate_compliance_parameters()

    def _validate_model_parameters(self) -> None:
        """Validate model architecture and training parameters."""
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                f"Must be one of: {', '.join(SUPPORTED_MODEL_TYPES)}"
            )

        if self.max_medical_seq_length <= 0:
            raise ValueError(
                "max_medical_seq_length must be positive, "
                f"got {self.max_medical_seq_length}"
            )

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.cache_ttl < 0:
            raise ValueError(f"cache_ttl must be non-negative, got {self.cache_ttl}")

    def _validate_medical_domain_parameters(self) -> None:
        """Validate medical domain-specific parameters."""
        # Validate medical specialties
        if not self.medical_specialties:
            warnings.warn(
                "No medical specialties specified. The model may not perform "
                "well "
                "without domain specialization.",
                UserWarning,
                stacklevel=2,
            )

        # Validate anatomical regions
        if not self.anatomical_regions:
            warnings.warn(
                "No anatomical regions specified. Entity recognition may be "
                "limited.",
                UserWarning,
                stacklevel=2,
            )

        # Validate imaging modalities
        if not self.imaging_modalities:
            warnings.warn(
                "No imaging modalities specified. Image-related features will "
                "be disabled.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_ner_parameters(self) -> None:
        """Validate named entity recognition parameters."""
        # Validate NER confidence threshold
        if not (0 <= self.ner_confidence_threshold <= 1.0):
            msg = (
                f"ner_confidence_threshold must be between 0 and 1, "
                f"got {self.ner_confidence_threshold}"
            )
            raise ValueError(msg)

        # Validate entity span length
        if (
            not isinstance(self.max_entity_span_length, int)
            or self.max_entity_span_length <= 0
        ):
            msg = (
                f"max_entity_span_length must be positive, "
                f"got {self.max_entity_span_length}"
            )
            raise ValueError(msg)

        # Validate entity types
        if not self.medical_entity_types:
            warnings.warn(
                "No entity types specified. NER functionality will be " "limited.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_performance_parameters(self) -> None:
        """Validate performance and resource-related parameters."""
        # Validate uncertainty threshold
        if not (0 <= self.uncertainty_threshold <= 1.0):
            msg = (
                f"uncertainty_threshold must be between 0 and 1, "
                f"got {self.uncertainty_threshold}"
            )
            raise ValueError(msg)

        # Validate domain adaptation parameters
        if self.domain_adaptation and not (0 <= self.domain_adaptation_lambda <= 1.0):
            msg = (
                "domain_adaptation_lambda must be between 0 and 1 when "
                "domain_adaptation is True, "
                f"got {self.domain_adaptation_lambda}"
            )
            raise ValueError(msg)

        # Validate request timeout
        if self.request_timeout <= 0:
            raise ValueError(
                f"request_timeout must be positive, got {self.request_timeout}"
            )

        # Validate max retries
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )

    def _validate_compliance_parameters(self) -> None:
        """Validate compliance and regulatory parameters."""
        if not self.regulatory_compliance:
            warnings.warn(
                "No regulatory compliance standards specified. Ensure this "
                "meets your organization's requirements.",
                UserWarning,
                stacklevel=2,
            )

        # Check for HIPAA compliance warning
        if "hipaa" not in [str(std).lower() for std in self.regulatory_compliance]:
            warnings.warn(
                "HIPAA compliance not specified. Ensure proper handling of "
                "protected health information (PHI).",
                UserWarning,
                stacklevel=2,
            )

    def _set_default_pretrained_paths(self) -> None:
        """Set default pretrained model paths if not specified.

        This method ensures that the pretrained_model_name_or_path is set to
        the model path if it hasn't been explicitly set.
        """
        if not self.pretrained_model_name_or_path and self.model:
            self.pretrained_model_name_or_path = self.model

    @classmethod
    def _convert_to_serializable(
        cls,
        value: Any,
        serialize_enums: bool = True,
        **kwargs: Any,
    ) -> Union[None, str, int, float, bool, List[Any], Dict[str, Any]]:
        """Recursively convert a value to a serializable format.

        This method handles conversion of various types to formats that can be
        serialized to JSON or YAML, including:
        - Basic types (None, str, int, float, bool)
        - Collections (list, tuple, set, dict)
        - Enums (converted to their values)
        - Objects with to_dict() methods
        - Objects with __dict__ attributes

        Args:
            value: The value to convert to a serializable format
            serialize_enums: If True, convert Enum values to their primitive values
            **kwargs: Additional keyword arguments for custom serialization

        Returns:
            A value that can be serialized to JSON or YAML

        Raises:
            TypeError: If the value cannot be converted to a serializable format
        """
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {
                str(k): cls._convert_to_serializable(v, serialize_enums, **kwargs)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                cls._convert_to_serializable(v, serialize_enums, **kwargs)
                for v in value
            ]

        if serialize_enums and isinstance(value, Enum):
            return value.value

        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            dict_value = value.to_dict()
            if not isinstance(dict_value, dict):
                raise TypeError(
                    f"to_dict() did not return a dictionary, got {type(dict_value).__name__}"
                )
            return cls._convert_to_serializable(dict_value, serialize_enums, **kwargs)

        if hasattr(value, "__dict__"):
            return cls._convert_to_serializable(vars(value), serialize_enums, **kwargs)

        try:
            return str(value)
        except Exception as e:
            raise TypeError(
                f"Cannot serialize value of type {type(value).__name__}: {str(e)}"
            ) from e

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> T:
        """Load a pretrained model configuration.

        Args:
            pretrained_model_name_or_path: Either:
                - A string, the model id of a pretrained model configuration hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a directory containing a configuration file saved using the `save_pretrained` method, e.g.,
                  `./my_model_directory/`.
            cache_dir: Optional directory to store the pre-trained model configurations downloaded from huggingface.co.
            force_download: Whether or not to force the (re-)download of the model weights and configuration files, overriding
                the cached versions if they exist.
            local_files_only: Whether or not to only look at local files (e.g., not try to download the model).
            **kwargs: Additional keyword arguments passed along to the base class method.

        Returns:
            An instance of the configuration class.

        Raises:
            OSError: If the configuration file cannot be found or is not a valid JSON file.
            ValueError: If the configuration file does not contain the required fields.
        """
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            **kwargs,
        )

        # Get the model type from the class if it exists, otherwise use the config dict
        model_type = getattr(cls, "model_type", None)
        if (
            model_type is not None
            and "model_type" in config_dict
            and config_dict["model_type"] != model_type
        ):
            warnings.warn(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{model_type}. This is not supported for all configurations of models and can produce errors.",
                UserWarning,
                stacklevel=2,
            )

        # Use the class's from_dict method if it exists, otherwise create a new instance
        if hasattr(cls, "from_dict") and callable(cls.from_dict):
            return cls.from_dict(config_dict, **kwargs)  # type: ignore[call-arg]
        return cls(**config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get the configuration dictionary from a pretrained model.

        Args:
            pretrained_model_name_or_path: Either:
                - A string, the model id of a pretrained model configuration hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a directory containing a configuration file saved using the `save_pretrained` method, e.g.,
                  `./my_model_directory/`.
            cache_dir: Optional directory to store the pre-trained model configurations downloaded from huggingface.co.
            force_download: Whether or not to force the (re-)download of the model weights and configuration files, overriding
                the cached versions if they exist.
            local_files_only: Whether or not to only look at local files (e.g., not try to download the model).
            **kwargs: Additional keyword arguments passed along to the base class method.

        Returns:
            A tuple containing the configuration dictionary and the keyword arguments.

        Raises:
            OSError: If the configuration file cannot be found or is not a valid JSON file.
            ValueError: If the configuration file does not contain the required fields.
        """
        config_dict: Dict[str, Any] = {}
        kwargs = kwargs or {}

        try:
            # Check if the path exists and is a directory
            if os.path.isdir(pretrained_model_name_or_path):
                # Try to load from JSON config file first
                json_config_file = os.path.join(
                    pretrained_model_name_or_path, "config.json"
                )
                if os.path.exists(json_config_file):
                    with open(json_config_file, "r", encoding="utf-8") as f:
                        config_dict = json.load(f)
                elif YAML_AVAILABLE:
                    # Fall back to YAML if JSON doesn't exist and YAML is available
                    yaml_config_file = os.path.join(
                        pretrained_model_name_or_path, "config.yaml"
                    )
                    if os.path.exists(yaml_config_file):
                        with open(yaml_config_file, "r", encoding="utf-8") as f:
                            config_dict = yaml.safe_load(f) or {}
                    else:
                        raise FileNotFoundError(
                            f"No config file found in {pretrained_model_name_or_path}. "
                            "Expected to find either config.json or config.yaml."
                        )

                # Process the configuration dictionary
                if not isinstance(config_dict, dict):
                    raise ValueError(
                        f"Configuration file should contain a dictionary, got {type(config_dict)}"
                    )

                # Ensure model_type is set if not already present
                if "model_type" not in config_dict and hasattr(cls, "model_type"):
                    config_dict["model_type"] = cls.model_type

                return config_dict, kwargs

            # If not a directory, try to load from Hugging Face Hub
            try:
                from huggingface_hub import hf_hub_download  # type: ignore[import]

                # First try to get the config file directly
                try:
                    config_file = hf_hub_download(
                        repo_id=str(pretrained_model_name_or_path),
                        filename="config.json",
                        cache_dir=str(cache_dir) if cache_dir else None,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                    with open(config_file, "r", encoding="utf-8") as f:
                        config_dict = json.load(f)
                except Exception as e:
                    if not YAML_AVAILABLE:
                        raise
                    # Fall back to YAML if JSON doesn't exist and YAML is available
                    try:
                        config_file = hf_hub_download(
                            repo_id=str(pretrained_model_name_or_path),
                            filename="config.yaml",
                            cache_dir=str(cache_dir) if cache_dir else None,
                            force_download=force_download,
                            local_files_only=local_files_only,
                        )
                        with open(config_file, "r", encoding="utf-8") as f:
                            config_dict = yaml.safe_load(f) or {}
                    except Exception as yaml_err:
                        raise FileNotFoundError(
                            f"Could not find or load config.json or config.yaml in {pretrained_model_name_or_path} "
                            f"on the Hugging Face Hub. Error: {str(yaml_err)}"
                        ) from yaml_err

                # Process the configuration dictionary
                if not isinstance(config_dict, dict):
                    raise ValueError(
                        f"Configuration file should contain a dictionary, got {type(config_dict)}"
                    )

                # Ensure model_type is set if not already present
                if "model_type" not in config_dict and hasattr(cls, "model_type"):
                    config_dict["model_type"] = cls.model_type

                return config_dict, kwargs

            except ImportError as import_err:
                raise ImportError(
                    "The `huggingface_hub` package is required to load models from the Hub. "
                    "Please install it with `pip install huggingface_hub`."
                ) from import_err

        except Exception as e:
            raise OSError(
                f"Error loading configuration from {pretrained_model_name_or_path}: {str(e)}"
            ) from e

        # Ensure we have a valid config dictionary
        if not isinstance(config_dict, dict):
            raise ValueError(
                f"Expected config to be a dictionary, got {type(config_dict).__name__}"
            )

        # Update with any overrides from kwargs
        if kwargs:
            config_dict.update(kwargs)

        # Convert the config dictionary to a config object
        return config_dict, kwargs

    def save_pretrained(
        self, save_directory: Union[str, os.PathLike], **kwargs: Any
    ) -> None:
        """Save the configuration to a directory.

        This method saves the configuration as JSON and YAML files in the
        specified directory. If the directory does not exist, it will be created.

        Args:
            save_directory: Directory path where the configuration files will be saved.
                Can be a string or a path-like object.
            **kwargs: Additional keyword arguments passed to the serialization methods.
                Common options include:
                - indent: int - Number of spaces for indentation in the output files
                - sort_keys: bool - Whether to sort dictionary keys in the output
                - ensure_ascii: bool - If False, non-ASCII characters are output as-is

        Raises:
            OSError: If the directory cannot be created or is not writable
            ValueError: If the save_directory is not a valid path
            TypeError: If the configuration cannot be serialized to JSON or YAML
            yaml.YAMLError: If there is an error during YAML serialization
        """
        save_dir = Path(save_directory).resolve()

        # Create the directory if it doesn't exist
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Unable to create directory {save_dir}: {str(e)}") from e

        # Verify the directory is writable
        if not os.access(save_dir, os.W_OK):
            raise OSError(
                f"Directory {save_dir} is not writable. "
                "Check permissions and try again."
            )

        # Get the configuration as a serializable dictionary
        try:
            config_dict = self.to_dict()
            serialized = self._convert_to_serializable(
                config_dict, serialize_enums=True
            )
            if not isinstance(serialized, dict):
                raise TypeError(
                    f"Serialized config must be a dictionary, got {type(serialized).__name__}"
                )
        except Exception as e:
            raise TypeError(
                f"Failed to convert configuration to dictionary: {str(e)}"
            ) from e

        # Save as JSON
        json_path = save_dir / "config.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    serialized,
                    f,
                    indent=kwargs.pop("indent", 2),
                    ensure_ascii=kwargs.pop("ensure_ascii", False),
                    sort_keys=kwargs.pop("sort_keys", True),
                    **kwargs,
                )
        except (TypeError, OverflowError) as e:
            raise TypeError(
                f"Failed to serialize configuration to JSON: {str(e)}"
            ) from e
        except OSError as e:
            raise OSError(
                f"Failed to write configuration to {json_path}: {str(e)}"
            ) from e

        # Save as YAML if available
        if YAML_AVAILABLE:
            yaml_path = save_dir / "config.yaml"
            try:
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(
                        serialized,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=kwargs.get("sort_keys", True),
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k not in ["indent", "ensure_ascii"]
                        },
                    )
            except yaml.YAMLError as e:
                # Don't fail if YAML serialization fails, just log a warning
                warnings.warn(
                    f"Failed to save YAML configuration: {str(e)}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        logger.info(f"Configuration saved in {save_dir}")


def to_yaml(
    self,
    file_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Optional[str]:
    """Serialize the configuration to a YAML string or file.

    This method converts the configuration to a YAML-formatted string. If a file path
    is provided, it will also save the YAML to that file.

    Args:
        file_path: Optional path where to save the YAML file. If None, the YAML
            string will be returned instead of being written to a file.
        **kwargs: Additional keyword arguments passed to `yaml.dump()`.
            Common options include:
            - default_flow_style: bool - If False, uses block style for better readability
            - sort_keys: bool - Whether to sort dictionary keys
            - width: int - Maximum line width
            - indent: int - Number of spaces for indentation

    Returns:
        The YAML string if file_path is None, otherwise None

    Raises:
        ImportError: If PyYAML is not installed
        ValueError: If the configuration cannot be serialized or file cannot be written
        TypeError: If the configuration contains unserializable types
        yaml.YAMLError: If there is an error during YAML serialization
        OSError: If there is an error writing to the file

    Example:
        ```python
        # Get YAML as a string
        yaml_str = config.to_yaml()

        # Save YAML to a file
        config.to_yaml("config.yaml")
        ```
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to use to_yaml(). "
            "Please install it with: pip install pyyaml"
        )

    try:
        # Convert the config to a serializable dictionary
        config_dict = self.to_dict()
        serialized = self._convert_to_serializable(config_dict)

        # Set default YAML serialization options if not provided
        yaml_kwargs = {
            "default_flow_style": False,
            "allow_unicode": True,
            "sort_keys": kwargs.pop("sort_keys", True),
            "width": kwargs.pop("width", 80),
            "indent": kwargs.pop("indent", 2),
        }
        yaml_kwargs.update(kwargs)  # Allow overriding defaults

        # Generate YAML string
        try:
            yaml_str = yaml.dump(serialized, **yaml_kwargs)
        except yaml.YAMLError as e:
            if hasattr(e, "problem_mark"):
                mark = e.problem_mark
                raise yaml.YAMLError(
                    f"YAML serialization error at position (line {mark.line + 1}, "
                    f"column {mark.column + 1}): {str(e)}"
                ) from e
            raise

        if not isinstance(yaml_str, str):
            raise ValueError("Failed to generate YAML string")

        # Write to file if path is provided
        if file_path is not None:
            file_path = Path(file_path)
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(yaml_str)
                return None
            except (IOError, OSError) as e:
                raise OSError(f"Failed to write YAML to {file_path}: {str(e)}") from e

        return yaml_str

    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize configuration to YAML: {str(e)}") from e
    except Exception as e:
        # Reraise any unhandled exceptions with additional context
        raise type(e)(f"Error in to_yaml(): {str(e)}") from e


def to_json(
    self,
    file_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Optional[str]:
    """Serialize the configuration to a JSON string or file.

    Args:
        file_path: Optional path to save the JSON file. If None, returns the JSON string.
        **kwargs: Additional keyword arguments passed to json.dump()

    Returns:
        The JSON string if file_path is None, otherwise None

    Raises:
        ValueError: If the configuration cannot be serialized or file cannot be written
        TypeError: If the configuration contains unserializable types
    """
    try:
        # Convert the config to a serializable dictionary
        config_dict = self.to_dict()
        serialized = self._convert_to_serializable(config_dict)

        # Default JSON serialization parameters
        json_kwargs = {
            "ensure_ascii": False,
            "indent": 2,
            "sort_keys": True,
        }
        json_kwargs.update(kwargs)  # Allow overriding defaults

        # Generate JSON string
        json_str = json.dumps(serialized, **json_kwargs)

        # Write to file if path is provided
        if file_path is not None:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
                return None
            except IOError as e:
                raise ValueError(
                    f"Failed to write JSON to {file_path}: {str(e)}"
                ) from e

        return json_str

    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize configuration to JSON: {str(e)}") from e


@classmethod
def _convert_dict_values(
    cls,
    config_dict: Dict[str, Any],
    field_type: Optional[Type[Any]] = None,
    field_name: str = "",
) -> Dict[str, Any]:
    """Recursively convert dictionary values to appropriate types.

    Args:
        config_dict: The dictionary to convert values in
        field_type: The expected type of the field (if known)
        field_name: Name of the field being processed (for error messages)

    Returns:
        The converted dictionary with proper types

    Raises:
        TypeError: If the input is not a dictionary
        ValueError: If a value cannot be converted to the expected type
    """
    if not isinstance(config_dict, dict):
        if config_dict is None:
            return {}
        raise TypeError(f"Expected a dictionary, got {type(config_dict).__name__}")

    converted: Dict[str, Any] = {}

    # Get the field type for the dictionary values if available
    value_type: Optional[Type[Any]] = None
    if field_type and hasattr(field_type, "__args__") and len(field_type.__args__) > 1:  # type: ignore
        value_type = field_type.__args__[1]  # type: ignore

    for key, value in config_dict.items():
        if not isinstance(key, str):
            warnings.warn(
                f"Non-string key '{key}' found in dictionary, converting to string",
                stacklevel=2,
            )
            str_key = str(key)
        else:
            str_key = key

        # Get the specific field type for this key if available
        key_field_type: Optional[Type[Any]] = None
        if field_type and hasattr(field_type, "__annotations__"):
            key_field_type = field_type.__annotations__.get(key)

        try:
            converted[str_key] = cls._convert_value(
                value, key, key_field_type or value_type, field_name
            )
        except (ValueError, TypeError) as e:
            warning_msg = (
                f"Could not convert {field_name}.{key}: {e}"
                if field_name
                else f"Could not convert {key}: {e}"
            )
            warnings.warn(warning_msg, stacklevel=2)
            converted[str_key] = value

    return converted


@classmethod
def _convert_non_dict_input(cls, obj: Any) -> Dict[str, Any]:
    """Helper method to handle non-dict input for _convert_dict_values.

    Args:
        obj: The input object to convert to a dictionary

    Returns:
        A dictionary representation of the input object

    Raises:
        TypeError: If the input cannot be converted to a dictionary
    """
    if obj is None:
        return {}
    if isinstance(obj, (str, int, float, bool)):
        return {"value": obj}
    if isinstance(obj, (list, tuple)):
        return {str(i): v for i, v in enumerate(obj)}
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        result = obj.to_dict()
        if not isinstance(result, dict):
            raise TypeError(
                f"to_dict() did not return a dictionary, got {type(result).__name__}"
            )
        return result
    return {"value": obj}


@classmethod
def _convert_value(
    cls,
    value: Any,
    key: str,
    field_type: Optional[Type[Any]] = None,
    field_name: str = "",
) -> Any:
    """Convert a single value to its appropriate type.

    Args:
        value: The value to convert
        key: The dictionary key this value belongs to
        field_type: The expected type of the field (if known)
        field_name: Name of the field being processed (for error messages)

    Returns:
        The converted value with the appropriate type

    Raises:
        ValueError: If the value cannot be converted to the target type
        TypeError: If the value is of an incompatible type
    """
    # If no type information is available, return as is
    if field_type is None:
        return value

    # Get the expected type for this field if available
    current_field_type: Optional[Type[Any]] = None
    if hasattr(field_type, "__annotations__"):
        current_field_type = field_type.__annotations__.get(key, field_type)
    else:
        current_field_type = field_type

    # Handle None values
    if value is None:
        return None

    # Handle nested dictionaries
    if isinstance(value, dict):
        nested_name = f"{field_name}.{key}" if field_name else key
        return cls._convert_dict_values(
            value,
            current_field_type,
            nested_name,
        )

    # Handle lists and tuples
    if isinstance(value, (list, tuple)):
        return cls._convert_list_value(value, current_field_type, field_name)

    # Handle enum types
    if (
        current_field_type
        and isinstance(value, str)
        and hasattr(current_field_type, "__members__")
    ):
        try:
            return current_field_type[value.upper()]
        except (KeyError, AttributeError):
            # If we can't convert to enum, try the string value as is
            pass

    # Handle boolean strings
    if current_field_type is bool and isinstance(value, str):
        normalized = value.lower().strip()
        if normalized in ("true", "1", "yes", "y"):
            return True
        if normalized in ("false", "0", "no", "n"):
            return False

    # Handle numeric strings
    if current_field_type in (int, float) and isinstance(value, str):
        try:
            return current_field_type(value)
        except (ValueError, TypeError):
            pass

    # Handle basic type conversion if the value isn't already the right type
    if current_field_type and not isinstance(value, current_field_type):
        try:
            return current_field_type(value)
        except (TypeError, ValueError) as e:
            error_msg = (
                f"Could not convert value '{value}' to type "
                f"{getattr(current_field_type, '__name__', str(current_field_type))}"
            )
            if field_name:
                error_msg += f" for field '{field_name}.{key}'"
            else:
                error_msg += f" for key '{key}'"
            raise ValueError(error_msg) from e

    return value


@classmethod
def _convert_list_value(
    cls,
    value: Any,
    field_name: str,
    field_type: Optional[Type[Any]] = None,
) -> List[Any]:
    """Convert a list value to the appropriate type.

        Args:
            value: The value to convert
            field_name: The name of the field being converted (for error messages)
            field_type: The expected type of the list elements

    Returns:
            The converted list

    Raises:
            ValueError: If the value cannot be converted to a list
            TypeError: If the list elements cannot be converted to the expected type
    """
    if value is None:
        return []

    # Handle string input
    if isinstance(value, str):
        try:
            # Try to parse as JSON
            parsed = json.loads(value)
            if isinstance(parsed, (list, tuple)):
                value = parsed
            else:
                value = [parsed]  # Single value wrapped in a list
        except json.JSONDecodeError:
            value = [value]  # Treat as single-element list

    # Handle non-sequence input
    if not isinstance(value, (list, tuple)):
        if field_type is not None and not isinstance(field_type, type):
            # Handle generic types like List[int]
            if hasattr(field_type, "__origin__") and field_type.__origin__ in (
                list,
                List,
            ):
                inner_type = field_type.__args__[0] if field_type.__args__ else Any
                return [cls._convert_value(value, f"{field_name}[0]", inner_type)]
        return [value]  # Wrap single value in a list

    # Get the inner type if this is a List[type] or list[type] annotation
    inner_type: Type[Any] = Any  # Default to Any if type can't be determined
    if field_type is not None:
        if hasattr(field_type, "__origin__") and field_type.__origin__ in (
            list,
            List,
            tuple,
            Tuple,
        ):
            if hasattr(field_type, "__args__") and field_type.__args__:
                inner_type = field_type.__args__[0]
        elif isinstance(field_type, type) and field_type in (list, List, tuple, Tuple):
            inner_type = Any

    # Convert each item in the list
    converted: List[Any] = []
    for i, item in enumerate(value):
        try:
            converted_item = cls._convert_value(item, f"{field_name}[{i}]", inner_type)
            converted.append(converted_item)
        except (ValueError, TypeError) as e:
            warnings.warn(f"Could not convert {field_name}[{i}]: {e}", stacklevel=2)
            converted.append(item)

    return converted
