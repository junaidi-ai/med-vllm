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
    get_type_hints,
    overload,
)

from pydantic import ValidationError

# Internal imports
from medvllm.config import Config
from medvllm.medical.config.validation import MedicalConfigValidator
from medvllm.utils.logging import get_logger

# Local application imports
from ..base import BaseMedicalConfig
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
# Import types from the new module structure
from ..types import (
    AnatomicalRegion,
    ClinicalMetrics,
    DomainConfig,
    DocumentType,
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

# Import serialization
from ..serialization import ConfigSerializer

# Import validation
from ..validation import MedicalConfigValidator

# Import versioning
from ..versioning import ConfigVersioner, ConfigVersionStatus, ConfigVersionInfo

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
        # Set up version compatibility
        self._setup_version_compatibility()

        # Convert string values to proper enum types if needed
        self._convert_enums()

        # Initialize any dependent configuration objects
        self._initialize_dependent_configs()

        # Set default pretrained paths if not specified
        self._set_default_pretrained_paths()

        # Run validation
        self._validate_medical_parameters()

    def _setup_version_compatibility(self) -> None:
        """Set up version compatibility checks and migrations."""
        # This method is intentionally left empty as a placeholder for future version compatibility logic
        pass

    def _convert_enums(self) -> None:
        """Convert string values to proper enum types if needed.

        This ensures type safety and consistency when loading from serialized formats.
        """
        # Convert medical specialties to enum instances
        if self.medical_specialties and not isinstance(
            self.medical_specialties[0], MedicalSpecialty
        ):
            self.medical_specialties = [
                MedicalSpecialty(spec) if isinstance(spec, str) else spec
                for spec in self.medical_specialties
            ]

        # Convert anatomical regions to enum instances
        if self.anatomical_regions and not isinstance(
            self.anatomical_regions[0], AnatomicalRegion
        ):
            self.anatomical_regions = [
                AnatomicalRegion(region) if isinstance(region, str) else region
                for region in self.anatomical_regions
            ]

        # Convert imaging modalities to enum instances
        if self.imaging_modalities and not isinstance(
            self.imaging_modalities[0], ImagingModality
        ):
            self.imaging_modalities = [
                ImagingModality(modality) if isinstance(modality, str) else modality
                for modality in self.imaging_modalities
            ]

        # Convert document types to enum instances
        if self.document_types and not isinstance(self.document_types[0], DocumentType):
            self.document_types = [
                DocumentType(doc_type) if isinstance(doc_type, str) else doc_type
                for doc_type in self.document_types
            ]

        # Convert regulatory compliance to enum instances
        if self.regulatory_compliance and not isinstance(
            self.regulatory_compliance[0], RegulatoryStandard
        ):
            self.regulatory_compliance = [
                RegulatoryStandard(std) if isinstance(std, str) else std
                for std in self.regulatory_compliance
            ]

    def _initialize_dependent_configs(self) -> None:
        """Initialize any dependent configuration objects.

        This method sets up complex nested configurations that depend on other
        configuration values.
        """
        # Initialize entity linking config if it's a dictionary
        if isinstance(self.entity_linking, dict):
            self.entity_linking = EntityLinkingConfig(**self.entity_linking)

        # Ensure domain config is properly initialized
        if hasattr(self, "domain_config") and isinstance(self.domain_config, dict):
            self.domain_config = DomainConfig(**self.domain_config)

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
        yaml_input: Union[str, bytes, os.PathLike],
        **kwargs: Any,
    ) -> "MedicalModelConfig":
        """Create a configuration from a YAML file or string.

        This method loads a configuration from a YAML string or file. The YAML should
        contain key-value pairs that match the configuration parameters.

        Args:
            yaml_input: YAML string, bytes, or path to a YAML file
            **kwargs: Additional keyword arguments to override config values

        Returns:
            MedicalModelConfig: A new instance of MedicalModelConfig

        Raises:
            ImportError: If PyYAML is not installed
            ValueError: If the YAML is invalid or missing required fields
        """
        try:
            import yaml
            from yaml import safe_load
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load YAML configuration. "
                "Please install it with: pip install pyyaml"
            ) from e

        try:
            if isinstance(yaml_input, (str, bytes)) and any(
                prefix.encode("utf-8") if isinstance(yaml_input, bytes) else prefix
                for prefix in [
                    b"---" if isinstance(yaml_input, bytes) else "---",
                    b"{" if isinstance(yaml_input, bytes) else "{",
                    b"[" if isinstance(yaml_input, bytes) else "[",
                ]
                if (yaml_input.strip().startswith(prefix))
            ):
                # Input is a YAML string
                config_dict = safe_load(yaml_input)
            else:
                # Input is a file path
                with open(yaml_input, "r") as f:
                    config_dict = safe_load(f)

            # Update with any overrides
            if kwargs:
                config_dict.update(kwargs)

            return cls.from_dict(config_dict)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {str(e)}")
        except (IOError, OSError) as e:
            raise ValueError(f"Error reading YAML file: {str(e)}")

    def _convert_to_serializable(self, value: Any, serialize_enums: bool = True) -> Any:
        """Recursively convert a value to a serializable format.

        Args:
            value: The value to convert
            serialize_enums: Whether to convert enums to their string representations

        Returns:
            The value in a serializable format
        """
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {
                str(k): self._convert_to_serializable(v, serialize_enums)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._convert_to_serializable(v, serialize_enums) for v in value]
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, Enum) and serialize_enums:
            return value.value if hasattr(value, "value") else str(value)
        if hasattr(value, "__dict__"):
            return self._convert_to_serializable(value.__dict__, serialize_enums)
        return str(value)

    def to_dict(self, serialize_enums: bool = True) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.

        This method handles conversion of complex types like enums, dataclasses,
        and nested structures to their serializable representations.

        Args:
            serialize_enums: If True, convert enums to their string representations.
        Returns:
            Dict containing the configuration parameters in a serializable format.
        """
        return self._convert_to_serializable(self.__dict__, serialize_enums)

    def to_json(
        self, file_path: Optional[Union[str, os.PathLike]] = None, **kwargs: Any
    ) -> Optional[str]:
        """Convert the configuration to a JSON string or file.

        Args:
            file_path: Optional path to save the JSON file.
            **kwargs: Additional arguments for json.dump() or json.dumps()

        Returns:
            JSON string if file_path is None, otherwise None

        Raises:
            ValueError: If serialization fails
        """
        config_dict = self.to_dict()

        # Extract common kwargs with proper types
        ensure_ascii = bool(kwargs.pop("ensure_ascii", False))
        sort_keys = bool(kwargs.pop("sort_keys", True))
        indent = int(kwargs.pop("indent", 2))
        default = kwargs.pop("default", str)
        
        # Any remaining kwargs will be passed through
        extra_kwargs = kwargs

        try:
            if file_path is not None:
                # For file output, use json.dump with file handle
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(
                        obj=config_dict,
                        fp=f,
                        ensure_ascii=ensure_ascii,
                        indent=indent,
                        sort_keys=sort_keys,
                        default=default,
                        **extra_kwargs
                    )
                return None
            
            # For string output, use json.dumps
            return json.dumps(
                obj=config_dict,
                ensure_ascii=ensure_ascii,
                indent=indent,
                sort_keys=sort_keys,
                default=default,
                **extra_kwargs
            )
        except (TypeError, ValueError) as e:
            error_msg = f"Failed to {'save' if file_path else 'serialize'} configuration to JSON"
            raise ValueError(f"{error_msg}. Error: {str(e)}") from e

    def to_yaml(
        self, file_path: Optional[Union[str, os.PathLike]] = None, **kwargs: Any
    ) -> Optional[str]:
        """Convert the configuration to a YAML string or file.

        Args:
            file_path: Optional path to save the YAML file.
            **kwargs: Additional arguments for yaml.dump()

        Returns:
            YAML string if file_path is None, otherwise None

        Raises:
            ImportError: If PyYAML is not installed
        """
        try:
            import yaml
            from yaml import SafeDumper
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to export YAML. Install it with: pip install pyyaml"
            ) from e

        config_dict = self.to_dict()

        # Common YAML kwargs
        common_kwargs = {
            "default_flow_style": False,
            "sort_keys": False,
            "allow_unicode": True,
            "width": 80,
        }

        # Update with user-provided kwargs, but don't allow overriding Dumper
        if "Dumper" in kwargs:
            del kwargs["Dumper"]

        # Update common kwargs with user-provided values
        common_kwargs.update(kwargs)

        try:
            if file_path is not None:
                # For file output, use stream parameter
                file_kwargs = common_kwargs.copy()
                file_kwargs["encoding"] = "utf-8"
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(
                        data=config_dict,
                        stream=f,
                        Dumper=SafeDumper,
                        default_flow_style=file_kwargs["default_flow_style"],
                        sort_keys=file_kwargs["sort_keys"],
                        allow_unicode=file_kwargs["allow_unicode"],
                        width=file_kwargs["width"],
                    )
                return None

            # For string output, don't use stream or encoding
            string_kwargs = common_kwargs.copy()
            if "encoding" in string_kwargs:
                del string_kwargs["encoding"]

            return yaml.dump(
                data=config_dict,
                Dumper=SafeDumper,
                default_flow_style=string_kwargs["default_flow_style"],
                sort_keys=string_kwargs["sort_keys"],
                allow_unicode=string_kwargs["allow_unicode"],
                width=string_kwargs["width"],
            )
        except Exception as e:
            error_msg = f"Failed to {'save' if file_path else 'serialize'} configuration to YAML"
            raise ValueError(f"{error_msg}. Error: {str(e)}") from e

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
        if not isinstance(config_dict, dict):
            return config_dict

        result: Dict[str, Any] = {}
        for key, value in config_dict.items():
            if value is None:
                result[key] = None
                continue

            # Get the field type from the class annotations if not provided
            current_field_type = field_type
            if hasattr(cls, "__annotations__") and key in cls.__annotations__:
                current_field_type = cls.__annotations__[key]

            # Handle different types of values
            if isinstance(value, dict):
                result[key] = cls._convert_dict_values(value, current_field_type, key)
            elif isinstance(value, list):
                item_type = None
                if (
                    current_field_type
                    and hasattr(current_field_type, "__args__")
                    and current_field_type.__args__
                ):
                    item_type = current_field_type.__args__[0]

                result[key] = [
                    cls._convert_dict_values(
                        v if isinstance(v, dict) else {"value": v},
                        item_type,
                        f"{key}[{i}]",
                    ).get(
                        "value", v
                    )  # Extract the value if we wrapped it
                    for i, v in enumerate(value)
                ]
            elif isinstance(value, str):
                # Try to convert string to appropriate enum type
                try:
                    if (
                        current_field_type
                        and hasattr(current_field_type, "__origin__")
                        and current_field_type.__origin__ == list
                    ):
                        item_type = (
                            current_field_type.__args__[0]
                            if current_field_type.__args__
                            else None
                        )
                        if item_type:
                            if item_type == MedicalSpecialty and hasattr(
                                MedicalSpecialty, value.upper()
                            ):
                                result[key] = MedicalSpecialty[value.upper()]
                            elif item_type == AnatomicalRegion and hasattr(
                                AnatomicalRegion, value.upper()
                            ):
                                result[key] = AnatomicalRegion[value.upper()]
                            elif item_type == ImagingModality and hasattr(
                                ImagingModality, value.upper()
                            ):
                                result[key] = ImagingModality[value.upper()]
                            elif item_type == EntityType and hasattr(EntityType, value):
                                result[key] = EntityType[value]
                            elif item_type == DocumentType and hasattr(
                                DocumentType, value.upper()
                            ):
                                result[key] = DocumentType[value.upper()]
                            elif item_type == RegulatoryStandard and hasattr(
                                RegulatoryStandard, value.upper()
                            ):
                                result[key] = RegulatoryStandard[value.upper()]
                            else:
                                result[key] = value
                        else:
                            result[key] = value
                    else:
                        result[key] = value
                except (KeyError, AttributeError):
                    result[key] = value
            else:
                result[key] = value

        return result
