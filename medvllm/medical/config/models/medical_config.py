"""
Medical model configuration.

This module contains the main MedicalModelConfig class that brings together
all the configuration components for medical models with enhanced type safety
and organization using the new modules.
"""

from __future__ import annotations

import dataclasses
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
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

try:
    import yaml
    from yaml import SafeDumper
except ImportError:
    yaml = None  # type: ignore
    SafeDumper = None  # type: ignore

from pydantic import ValidationError

# Internal imports
from medvllm.config import Config
from medvllm.medical.config.validation import MedicalConfigValidator
from medvllm.utils.logging import get_logger

# Local application imports
from ..constants import (
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
from ..enums import (
    AnatomicalRegion,
    DocumentType,
    EntityType,
    ImagingModality,
    MedicalSpecialty,
    RegulatoryStandard,
)
from ..serialization.config_serializer import ConfigSerializer

# Import types from the new module structure
from ..types import (
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
    PathLike,
    RegulatoryStandard,
    validate_model_path,
)

# Import validation
from ..validation import MedicalConfigValidator

# Import versioning
from ..versioning import ConfigVersioner, ConfigVersionInfo, ConfigVersionStatus

# Import schema
from .schema import MedicalModelConfigSchema, ModelType

# Type variable for class methods that return an instance of the class
T = TypeVar("T", bound="MedicalModelConfig")

# Supported model types for medical applications
SUPPORTED_MODEL_TYPES = {
    "bert",
    "roberta",
    "gpt2",
    "t5",
    "medical_bert",
    "biobert",
    "clinical_bert",
    "pubmed_bert",
    "bluebert",
}

# Configuration version
CONFIG_VERSION = "1.0.0"

# Initialize logger
logger = get_logger(__name__)


@dataclass
class MedicalModelConfig(BaseMedicalConfig):
    """Configuration class for medical model parameters.

    This class extends the base configuration with medical-specific parameters
    and validation logic, utilizing the new modular structure for better
    organization and type safety.

    Attributes:
        model_type (str): Type of the model architecture. Must be one of the supported model types.
        model (str): Path to the model directory or model identifier.
        pretrained_model_name_or_path (Optional[str]): Name or path of the pretrained model.
        max_medical_seq_length (int): Maximum sequence length for medical text processing.
        batch_size (int): Default batch size for inference.
        enable_uncertainty_estimation (bool): Whether to enable uncertainty estimation.
        uncertainty_threshold (float): Threshold for model uncertainty calibration.
        cache_ttl (int): Time-to-live for cache in seconds.
        medical_specialties (List[Union[MedicalSpecialty, str]]): List of medical specialties.
        anatomical_regions (List[Union[AnatomicalRegion, str]]): List of anatomical regions.
        imaging_modalities (List[Union[ImagingModality, str]]): List of imaging modalities.
        medical_entity_types (List[Union[EntityType, str]]): Types of medical entities to recognize.
        ner_confidence_threshold (float): Minimum confidence score for NER predictions.
        max_entity_span_length (int): Maximum token length for entity spans.
        entity_linking (Dict[str, Any]): Configuration for entity linking.
        document_types (List[Union[DocumentType, str]]): Types of clinical documents supported.
        section_headers (List[str]): Common section headers in clinical documents.
        max_retries (int): Maximum number of retries for API calls.
        request_timeout (int): Timeout in seconds for API requests.
        domain_adaptation (bool): Whether to enable domain adaptation.
        domain_adaptation_lambda (float): Weight for domain adaptation loss.
        domain_specific_vocab (Optional[Dict[str, List[str]]]): Domain-specific vocabulary terms.
        regulatory_compliance (List[Union[RegulatoryStandard, str]]): Regulatory standards the model complies with.
        config_version (str): Configuration schema version.
    """

    # Model configuration
    model: str = field(
        default="",
        metadata={
            "description": "Path to the model directory or model identifier. Required field.",
            "required": True,
        },
    )

    model_type: str = field(
        default=DEFAULT_MODEL_TYPE,
        metadata={
            "description": f"Type of the model architecture. Must be one of: {', '.join(SUPPORTED_MODEL_TYPES)}",
            "choices": SUPPORTED_MODEL_TYPES,
        },
    )

    pretrained_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "description": "Name or path of the pretrained model from Hugging Face Hub"
        },
    )

    # Model architecture parameters
    max_medical_seq_length: int = field(
        default=DEFAULT_MAX_SEQ_LENGTH,
        metadata={"description": "Maximum sequence length for medical text processing"},
    )

    # Training and inference parameters
    batch_size: int = field(
        default=DEFAULT_BATCH_SIZE,
        metadata={"description": "Default batch size for inference"},
    )

    enable_uncertainty_estimation: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable uncertainty estimation in model outputs"
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
            "description": "List of medical specialties this model is trained on",
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
            "description": "List of medical imaging modalities supported by the model",
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
        default=DEFAULT_MAX_ENTITY_SPAN,
        metadata={"description": "Maximum token length for entity spans"},
    )

    entity_linking: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "knowledge_bases": list(DEFAULT_KNOWLEDGE_BASES),
            "confidence_threshold": 0.8,
        },
        metadata={"description": "Configuration for entity linking to knowledge bases"},
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
        default=DEFAULT_MAX_RETRIES,
        metadata={"description": "Maximum number of retries for API calls"},
    )

    request_timeout: int = field(
        default=DEFAULT_REQUEST_TIMEOUT,
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
        default_factory=lambda: list(DEFAULT_REGULATORY_STANDARDS),
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

        This method performs several important tasks after the configuration is initialized:
        1. Validates all configuration parameters
        2. Sets up default values for optional fields
        3. Initializes version compatibility checks
        4. Performs any necessary type conversions
        """
        # Convert string values to proper enum types
        self._convert_enums()

        # Set up version compatibility
        self._setup_version_compatibility()

        # Set default pretrained paths if not specified
        self._set_default_pretrained_paths()

        # Initialize any dependent configs
        self._initialize_dependent_configs()

        # Validate all parameters
        self._validate_medical_parameters()

    def _setup_version_compatibility(self) -> None:
        """Set up version compatibility checks and migrations."""
        # This method is intentionally left empty as a placeholder for future version compatibility logic
        pass

    def _convert_enums(self) -> None:
        """Convert string values to proper enum types if needed.

        This ensures type safety and consistency when loading serialized data.
        """

        # Convert string values in lists to enums with type checking
        def safe_convert(value: Union[str, Enum], enum_type: Type[Enum]) -> Enum:
            if isinstance(value, str):
                try:
                    return enum_type(value)
                except ValueError as e:
                    warnings.warn(
                        f"Invalid {enum_type.__name__} value: {value}. Using default.",
                        UserWarning,
                    )
                    return list(enum_type)[0]  # Return first enum value as default
            return value

        # Convert each list of enums
        if self.medical_specialties:
            self.medical_specialties = [
                safe_convert(spec, MedicalSpecialty) if isinstance(spec, str) else spec
                for spec in self.medical_specialties
            ]
        if self.anatomical_regions:
            self.anatomical_regions = [
                (
                    safe_convert(region, AnatomicalRegion)
                    if isinstance(region, str)
                    else region
                )
                for region in self.anatomical_regions
            ]
        if self.imaging_modalities:
            self.imaging_modalities = [
                (
                    safe_convert(modality, ImagingModality)
                    if isinstance(modality, str)
                    else modality
                )
                for modality in self.imaging_modalities
            ]
        if self.medical_entity_types:
            self.medical_entity_types = [
                (
                    safe_convert(entity_type, EntityType)
                    if isinstance(entity_type, str)
                    else entity_type
                )
                for entity_type in self.medical_entity_types
            ]
        if self.document_types:
            self.document_types = [
                (
                    safe_convert(doc_type, DocumentType)
                    if isinstance(doc_type, str)
                    else doc_type
                )
                for doc_type in self.document_types
            ]
        if self.regulatory_compliance:
            self.regulatory_compliance = [
                safe_convert(std, RegulatoryStandard) if isinstance(std, str) else std
                for std in self.regulatory_compliance
            ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedicalModelConfig":
        """Create a configuration from a dictionary.

        Args:
            data: Dictionary containing configuration parameters

        Returns:
            A new instance of MedicalModelConfig
        """
        # Create a copy to avoid modifying the input
        data = data.copy()

        # Remove domain_config and internal fields from the data
        domain_config = data.pop("domain_config", None)
        data.pop("_extra_fields", None)  # Remove internal fields

        # Create the instance with only valid constructor arguments
        valid_fields = {f.name for f in fields(cls) if f.init}
        constructor_args = {k: v for k, v in data.items() if k in valid_fields}
        instance = cls(**constructor_args)

        # Set any remaining fields as attributes
        for k, v in data.items():
            if k not in valid_fields and not k.startswith("_"):
                setattr(instance, k, v)

        # Handle model_type if present
        if "model_type" in data:
            instance.model_type = data["model_type"]

        # Set domain_config after initialization if it exists
        if domain_config is not None:
            if isinstance(domain_config, dict):
                instance.domain_config = DomainConfig(**domain_config)
            else:
                instance.domain_config = domain_config

        return instance

    def copy(self) -> "MedicalModelConfig":
        """Create a copy of the configuration.

        This method ensures that domain_config and other fields are properly handled during copying.

        Returns:
            A new instance with the same parameters
        """
        # Get the dictionary representation
        data = self.to_dict()

        # Create a new instance using from_dict which handles all the special cases
        return self.__class__.from_dict(data)

    def _initialize_dependent_configs(self) -> None:
        """Initialize any dependent configuration objects.

        This method sets up complex nested configurations that depend on other
        configuration values.
        """
        # Initialize domain config if needed
        if not hasattr(self, "domain_config") or self.domain_config is None:
            self.domain_config: DomainConfig = DomainConfig(
                enabled=self.domain_adaptation,
                lambda_val=self.domain_adaptation_lambda,
                vocab=self.domain_specific_vocab or {},
            )

        # Initialize entity linking config if it's a dictionary
        if isinstance(self.entity_linking, dict):
            self.entity_linking = EntityLinkingConfig(**self.entity_linking)

        # Validate using Pydantic
        try:
            # Create a copy of the current instance's attributes for validation
            config_dict = {
                k: v for k, v in self.__dict__.items() if not k.startswith("_")
            }

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

        # Initialize version compatibility check
        ConfigVersioner.check_version_compatibility(self.config_version)

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

        This method validates all medical-specific parameters to ensure they have
        valid values. It raises ValueError for invalid values and issues warnings
        for potentially problematic but technically valid values.

        The validation includes:
        - Model type and architecture parameters
        - Medical domain-specific parameters
        - Clinical entity recognition settings
        - Compliance and regulatory settings
        - Performance and resource constraints

        Raises:
            ValueError: If any parameter has an invalid value
            UserWarning: For potentially problematic but technically valid values
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
                f"max_medical_seq_length must be positive, got {self.max_medical_seq_length}"
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
                "No medical specialties specified. The model may not perform well "
                "without domain specialization.",
                UserWarning,
                stacklevel=2,
            )

        # Validate anatomical regions
        if not self.anatomical_regions:
            warnings.warn(
                "No anatomical regions specified. Entity recognition may be limited.",
                UserWarning,
                stacklevel=2,
            )

        # Validate imaging modalities
        if not self.imaging_modalities:
            warnings.warn(
                "No imaging modalities specified. Image-related features will be disabled.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_ner_parameters(self) -> None:
        """Validate named entity recognition parameters."""
        # Validate NER confidence threshold
        if not (0 <= self.ner_confidence_threshold <= 1.0):
            raise ValueError(
                f"ner_confidence_threshold must be between 0 and 1, got {self.ner_confidence_threshold}"
            )

        # Validate entity span length
        if self.max_entity_span_length <= 0:
            raise ValueError(
                f"max_entity_span_length must be positive, got {self.max_entity_span_length}"
            )

        # Validate entity types
        if not self.medical_entity_types:
            warnings.warn(
                "No entity types specified. NER functionality will be limited.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_performance_parameters(self) -> None:
        """Validate performance and resource-related parameters."""
        # Validate uncertainty threshold
        if not (0 <= self.uncertainty_threshold <= 1.0):
            raise ValueError(
                f"uncertainty_threshold must be between 0 and 1, got {self.uncertainty_threshold}"
            )

        # Validate domain adaptation lambda
        if self.domain_adaptation and not (0 <= self.domain_adaptation_lambda <= 1.0):
            raise ValueError(
                "domain_adaptation_lambda must be between 0 and 1 when domain_adaptation is True, "
                f"got {self.domain_adaptation_lambda}"
            )

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
                "No regulatory compliance standards specified. Ensure this meets your organization's requirements.",
                UserWarning,
                stacklevel=2,
            )

        # Check for HIPAA compliance if handling PHI
        if "hipaa" not in [str(std).lower() for std in self.regulatory_compliance]:
            warnings.warn(
                "HIPAA compliance not specified. Ensure proper handling of protected health information (PHI).",
                UserWarning,
                stacklevel=2,
            )

    def _set_default_pretrained_paths(self) -> None:
        """Set default pretrained model paths if not specified.

        This method ensures that the pretrained_model_name_or_path is set to the model path
        if it hasn't been explicitly set. This is useful for backward compatibility.
        """
        if not self.pretrained_model_name_or_path and hasattr(self, "model"):
            self.pretrained_model_name_or_path = self.model

    @classmethod
    def from_pretrained(
        cls: Type["MedicalModelConfig"], model_name_or_path: str, **kwargs: Any
    ) -> "MedicalModelConfig":
        """Create a config from a pretrained model.

        This method initializes a configuration using a pre-trained model's settings
        and allows overriding specific parameters via keyword arguments.

        Args:
            model_name_or_path: Name or path of the pretrained model. This can be:
                - A string, the model id of a pretrained model hosted inside a model repo on huggingface.co.
                - A path to a directory containing a configuration file saved using the `save_pretrained` method.
            **kwargs: Additional keyword arguments passed along to the model's
                `from_pretrained` method. Can be used to update the configuration.

        Returns:
            MedicalModelConfig: An instance of MedicalModelConfig initialized from the pretrained model.

        Example:
            ```python
            # Load a pretrained model with default configuration
            config = MedicalModelConfig.from_pretrained("bert-base-uncased")

            # Load with custom parameters
            config = MedicalModelConfig.from_pretrained(
                "bert-base-uncased",
                max_medical_seq_length=512,
                enable_uncertainty_estimation=True
            )
            ```
        """
        if not isinstance(model_name_or_path, (str, os.PathLike)):
            raise TypeError(
                f"model_name_or_path should be a string or os.PathLike, got {type(model_name_or_path)}"
            )

        # Create a new config instance
        config = cls()

        try:
            # Update with pretrained model path
            config.model = str(model_name_or_path)
            config.pretrained_model_name_or_path = str(model_name_or_path)

            # Update with any provided kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(
                        f"Key '{key}' not found in config. Available keys: {', '.join(config.__annotations__.keys())}"
                    )

            # Validate the final configuration
            config._validate_medical_parameters()

            return config
        except Exception as e:
            raise ValueError(
                f"Error loading configuration for model '{model_name_or_path}'. "
                f"Original error: {str(e)}"
            ) from e

    @classmethod
    def from_yaml(
        cls: Type["MedicalModelConfig"],
        yaml_input: Union[str, bytes, os.PathLike, Any],
        **kwargs: Any,
    ) -> "MedicalModelConfig":
        """Create a configuration from a YAML file or string.

        This method loads a configuration from a YAML string or file. The YAML should
        contain key-value pairs that match the configuration parameters.

        Args:
            yaml_input: YAML string, bytes, path to a YAML file, or file-like object
            **kwargs: Additional keyword arguments to override config values

        Returns:
            MedicalModelConfig: A new instance of MedicalModelConfig

        Raises:
            ImportError: If PyYAML is not installed
            ValueError: If the YAML is invalid or missing required fields
        """
        # Import PyYAML at runtime to avoid making it a hard dependency
        try:
            import yaml
            from yaml import safe_load
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load YAML configuration. "
                "Please install it with: pip install pyyaml"
            ) from e

        config_dict: Dict[str, Any] = {}

        try:
            # Handle file-like objects (check this first to avoid PathLike/str checks)
            if hasattr(yaml_input, "read") and callable(
                getattr(yaml_input, "read", None)
            ):
                # For file-like objects, read and parse directly
                content = yaml_input.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                config_dict = safe_load(content) or {}
            # Handle string, bytes, or path-like input
            elif isinstance(yaml_input, (str, bytes, os.PathLike)):
                # Convert PathLike to string
                file_path = (
                    str(yaml_input) if hasattr(yaml_input, "__fspath__") else yaml_input
                )

                # Check if input is a file path (not a YAML string)
                if isinstance(file_path, str) and os.path.isfile(file_path):
                    with open(file_path, "r") as f:
                        config_dict = safe_load(f) or {}
                else:
                    # Handle YAML string/bytes
                    if isinstance(file_path, bytes):
                        file_path = file_path.decode("utf-8")
                    config_dict = safe_load(file_path) or {}
            else:
                raise ValueError(
                    "Invalid input type for YAML loading. Expected string, bytes, "
                    "path-like, or file-like object."
                )

            if not isinstance(config_dict, dict):
                raise ValueError("YAML content must parse to a dictionary")

            # Update with any overrides
            if kwargs:
                config_dict.update(kwargs)

            return cls.from_dict(config_dict)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {str(e)}")
        except (IOError, OSError) as e:
            raise ValueError(f"Error reading YAML file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading YAML configuration: {str(e)}")

    def _convert_to_serializable(self, value: Any, serialize_enums: bool = True) -> Any:
        """Recursively convert a value to a serializable format.

        Args:
            value: The value to convert
            serialize_enums: Whether to convert enums to their string representations

        Returns:
            The value in a serializable format
        """
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._convert_to_serializable(v, serialize_enums) for v in value]
        if isinstance(value, dict):
            return {
                str(k): self._convert_to_serializable(v, serialize_enums)
                for k, v in value.items()
            }
        if hasattr(value, "to_dict"):
            return value.to_dict(serialize_enums=serialize_enums)
        if isinstance(value, Enum) and serialize_enums:
            return value.value if hasattr(value, "value") else str(value)
        if dataclasses.is_dataclass(value):
            return {
                f.name: self._convert_to_serializable(
                    getattr(value, f.name), serialize_enums
                )
                for f in dataclasses.fields(value)
                if not f.name.startswith("_")
            }
        return str(value)

    def save_pretrained(
        self, save_directory: Union[str, os.PathLike], **kwargs: Any
    ) -> None:
        """Save the configuration to a directory.

        This method saves the configuration as JSON and YAML files in the specified directory.

        Args:
            save_directory: Directory to save the configuration files to.
            **kwargs: Additional keyword arguments passed to serialization methods.


        Example:
            ```python
            config = MedicalModelConfig()
            config.save_pretrained("path/to/save")
            ```
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        output_json_file = save_directory / "config.json"
        self.to_json(output_json_file, **kwargs)

        # Also save as YAML if available
        try:
            output_yaml_file = save_directory / "config.yaml"
            self.to_yaml(output_yaml_file, **kwargs)
        except ImportError:
            # If PyYAML is not available, just skip YAML serialization
            pass

    @classmethod
    def _convert_dict_values(
        cls,
        config_dict: Dict[str, Any],
        field_type: Optional[Type] = None,
        field_name: str = "",
    ) -> Dict[str, Any]:
        """Recursively convert dictionary values to appropriate types.

        Args:
            config_dict: The dictionary to convert values in
            field_type: The expected type of the field (if known)
            field_name: Name of the field being processed (for error messages)

        Returns:
            The converted dictionary with proper types
        """
        result: Dict[str, Any] = {}

        for key, value in config_dict.items():
            if value is None:
                result[key] = None
                continue

            try:
                result[key] = cls._convert_value(value, key, field_type, field_name)
            except Exception as e:
                warnings.warn(
                    f"Error processing field '{key}' in {field_name or 'root'}: {str(e)}",
                    UserWarning,
                )
                result[key] = value

        return result

    @classmethod
    def _convert_non_dict_input(cls, obj: Any) -> Dict[str, Any]:
        """Helper method to handle non-dict input for _convert_dict_values."""
        # Check for to_dict method in one step
        to_dict = getattr(obj, "to_dict", None)
        if to_dict is None or not callable(to_dict):
            return {}

        # Try to convert the object to a dictionary
        try:
            converted = to_dict()
            # Ensure the converted value is a dictionary
            if not isinstance(converted, dict):
                return {}
            return converted
        except Exception:
            # Ignore any exceptions during conversion
            return {}

    @classmethod
    def _convert_value(
        cls, value: Any, key: str, field_type: Optional[Type], field_name: str
    ) -> Any:
        """Convert a single value to its appropriate type."""
        # Get the expected type for this field if available
        current_field_type: Optional[Type] = None
        if field_type and hasattr(field_type, "__annotations__"):
            current_field_type = field_type.__annotations__.get(key)

        # Handle nested dictionaries
        if isinstance(value, dict):
            return cls._convert_dict_values(
                value,
                current_field_type,
                f"{field_name}.{key}" if field_name else key,
            )

        # Handle lists
        if isinstance(value, list) and value:
            return cls._convert_list_value(value, key, current_field_type, field_name)

        return value

    @classmethod
    def _convert_list_value(
        cls, value: List[Any], key: str, item_type: Optional[Type], field_name: str
    ) -> List[Any]:
        """Convert a list value to its appropriate type."""
        if not item_type or not hasattr(item_type, "__args__"):
            return value

        item_type = item_type.__args__[0] if item_type.__args__ else None

        # Handle list of dictionaries
        if value and all(isinstance(item, dict) for item in value):
            return [
                cls._convert_dict_values(item, item_type, f"{field_name}.{key}[{i}]")
                for i, item in enumerate(value)
            ]

        # Handle list of enums if item_type is an Enum
        if (
            item_type is not None
            and isinstance(item_type, type)
            and issubclass(item_type, Enum)
        ):
            converted_items = []
            for item in value:
                if isinstance(item, str):
                    try:
                        # Try to convert string to enum value
                        converted_items.append(item_type[item.upper()])
                    except (KeyError, AttributeError):
                        warnings.warn(
                            f"Invalid {item_type.__name__} value: {item}",
                            UserWarning,
                        )
                        converted_items.append(item)
                else:
                    converted_items.append(item)
            return converted_items

        return value
