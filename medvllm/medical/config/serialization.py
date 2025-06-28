"""
Serialization and deserialization utilities for medical model configurations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar, Union

from .base import BaseMedicalConfig

if TYPE_CHECKING:
    from .medical_config import MedicalModelConfig

T = TypeVar("T", bound="MedicalModelConfig")


class ConfigSerializer:
    """Handles serialization and deserialization of configuration objects."""

    @classmethod
    def to_dict(cls, config: BaseMedicalConfig) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        output = {}

        # Get base config parameters
        if hasattr(super(BaseMedicalConfig, config), "to_dict"):
            base_dict = super(BaseMedicalConfig, config).to_dict()
            if isinstance(base_dict, dict):
                output.update(base_dict)

        # Add medical-specific fields
        medical_fields = {
            "config_version": config.config_version,
            "model": getattr(config, "model", None),
            "model_type": getattr(config, "model_type", None),
            "medical_specialties": getattr(config, "medical_specialties", None),
            "anatomical_regions": getattr(config, "anatomical_regions", None),
            "imaging_modalities": getattr(config, "imaging_modalities", None),
            "clinical_metrics": getattr(config, "clinical_metrics", None),
            "regulatory_compliance": getattr(config, "regulatory_compliance", None),
            "use_crf": getattr(config, "use_crf", None),
            "do_lower_case": getattr(config, "do_lower_case", None),
            "preserve_case_for_abbreviations": getattr(
                config, "preserve_case_for_abbreviations", None
            ),
            "domain_adaptation": getattr(config, "domain_adaptation", None),
            "domain_adaptation_lambda": getattr(
                config, "domain_adaptation_lambda", None
            ),
            "domain_specific_vocab": getattr(config, "domain_specific_vocab", None),
            "pretrained_model_name_or_path": getattr(
                config, "pretrained_model_name_or_path", None
            ),
            "medical_vocab_file": getattr(config, "medical_vocab_file", None),
            "medical_entity_types": getattr(config, "medical_entity_types", None),
            "ner_confidence_threshold": getattr(
                config, "ner_confidence_threshold", None
            ),
            "max_entity_span_length": getattr(config, "max_entity_span_length", None),
            "entity_linking_enabled": getattr(config, "entity_linking_enabled", None),
            "entity_linking_knowledge_bases": getattr(
                config, "entity_linking_knowledge_bases", None
            ),
            "document_types": getattr(config, "document_types", None),
            "section_headers": getattr(config, "section_headers", None),
            "enable_uncertainty_estimation": getattr(
                config, "enable_uncertainty_estimation", None
            ),
            "uncertainty_threshold": getattr(config, "uncertainty_threshold", None),
            "max_retries": getattr(config, "max_retries", None),
            "request_timeout": getattr(config, "request_timeout", None),
            "batch_size": getattr(config, "batch_size", None),
            "enable_caching": getattr(config, "enable_caching", None),
            "cache_ttl": getattr(config, "cache_ttl", None),
            "max_cache_size": getattr(config, "max_cache_size", None),
        }

        # Only include non-None values
        output.update({k: v for k, v in medical_fields.items() if v is not None})
        return output

    @classmethod
    def to_json(
        cls,
        config: BaseMedicalConfig,
        file_path: Optional[Union[str, os.PathLike]] = None,
        indent: int = 2,
    ) -> Optional[str]:
        """Convert configuration to a JSON string or file."""
        config_dict = cls.to_dict(config)

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=indent, ensure_ascii=False)
            return None
        return json.dumps(config_dict, indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, config_class: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create a configuration from a dictionary with improved type handling.

        Args:
            config_class: The configuration class to instantiate
            config_dict: Dictionary containing configuration parameters

        Returns:
            An instance of config_class initialized with the provided parameters

        Raises:
            ValueError: If the configuration is invalid or missing required fields
            TypeError: If there are type mismatches in the configuration
        """
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict, got {type(config_dict).__name__}")

        config_dict = config_dict.copy()

        # Handle config version with type checking
        config_version = config_dict.pop("config_version", "0.1.0")
        if not isinstance(config_version, str):
            raise TypeError(
                f"config_version must be a string, got {type(config_version).__name__}"
            )

        try:
            # Create config instance with type checking
            config = config_class(**config_dict)

            # Set version and migrate if needed
            if (
                hasattr(config, "config_version")
                and config_version != config.config_version
            ):
                config.config_version = config_version
                from .versioning import ConfigVersioner

                ConfigVersioner.migrate_config(config)

            return config

        except TypeError as e:
            # Improve error message for type errors
            param = (
                str(e).split("'")[1] if "unexpected keyword argument" in str(e) else ""
            )
            if param:
                raise TypeError(f"Invalid type for parameter '{param}': {e}") from e
            raise

        except Exception as e:
            raise ValueError(f"Failed to create configuration: {str(e)}") from e

    @classmethod
    def from_json(
        cls, config_class: Type[T], json_input: Union[str, os.PathLike, Dict]
    ) -> T:
        """Create a configuration from a JSON string, file, or dictionary."""
        config_dict: Dict[str, Any]
        if isinstance(json_input, dict):
            config_dict = json_input
        elif isinstance(json_input, (str, os.PathLike)):
            json_str = str(json_input)  # Convert PathLike to string
            if os.path.isfile(json_str):
                with open(json_str, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
            else:
                config_dict = json.loads(json_str)
        else:
            raise TypeError("Input must be a JSON string, file path, or dictionary")

        return cls.from_dict(config_class, config_dict)
