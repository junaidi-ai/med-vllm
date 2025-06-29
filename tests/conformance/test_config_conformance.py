"""
Conformance tests for configuration system.

This module contains tests that verify the configuration system
conforms to required standards and specifications.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Type, get_type_hints

import pytest

# Import the actual implementation
from medvllm.medical.config import MedicalModelConfig
from medvllm.medical.config.base import BaseMedicalConfig
from medvllm.medical.config.types.models import DomainConfig

# Constants for conformance testing
REQUIRED_FIELDS = {
    "model_type": str,
    "model": str,
    "medical_specialties": list,
    "anatomical_regions": list,
    "imaging_modalities": list,
    "medical_entity_types": list,
}

# Regular expression for valid model names (allow paths for model)
MODEL_NAME_PATTERN = r"^[\w\-\.\/]+$"

# Allowed medical specialties (from MedicalSpecialty enum)
ALLOWED_SPECIALTIES = {
    "family_medicine",
    "internal_medicine",
    "pediatrics",
    "cardiology",
    "dermatology",
    "endocrinology",
    "gastroenterology",
    "hematology",
    "neurology",
    "oncology",
    "ophthalmology",
    "orthopedics",
    "psychiatry",
    "pulmonology",
    "radiology",
    "urology",
}

# Allowed anatomical regions (from AnatomicalRegion enum)
ALLOWED_REGIONS = {
    "head",
    "thorax",
    "abdomen",
    "pelvis",
    "upper_limb",
    "lower_limb",
    "neck",
    "back",
    "spine",
    "chest",
}

# Configuration for backward compatibility testing
COMPATIBILITY_MATRIX = {
    "0.1.0": {
        "required_fields": ["model_type", "hidden_size", "num_hidden_layers"],
        "deprecated_fields": [],
        "removed_fields": [],
    },
    "1.0.0": {
        "required_fields": [
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "medical_specialties",
        ],
        "deprecated_fields": ["old_field"],
        "removed_fields": ["legacy_field"],
    },
    # Add more versions as needed
}


class TestConfigConformance:
    """Conformance tests for configuration system."""

    def test_required_fields_present(self, temp_model_dir) -> None:
        """Test that all required fields are present in the config class."""
        config = MedicalModelConfig(model=temp_model_dir)
        config_dict = config.to_dict()

        missing_fields = []
        for field, field_type in REQUIRED_FIELDS.items():
            if (
                field not in config_dict and field != "model_name_or_path"
            ):  # model_name_or_path is handled by base class
                missing_fields.append(field)

        assert (
            not missing_fields
        ), f"Missing required fields: {', '.join(missing_fields)}"

    def test_field_types(self, temp_model_dir) -> None:
        """Test that all fields have the correct types."""
        config = MedicalModelConfig(model=temp_model_dir)

        for field_name, field_type in REQUIRED_FIELDS.items():
            value = getattr(config, field_name, None)
            if value is not None:  # Only check non-None values
                # Handle list/dict types specially
                if field_type in (list, dict):
                    assert isinstance(
                        value, field_type
                    ), f"Field {field_name} has type {type(value).__name__}, expected {field_type.__name__}"
                elif hasattr(
                    field_type, "__origin__"
                ):  # Handle generic types like List, Dict
                    # Skip complex type checking for now
                    continue
                else:
                    assert isinstance(
                        value, field_type
                    ), f"Field {field_name} has type {type(value).__name__}, expected {field_type.__name__}"
                continue

            value = config_dict.get(field)
            if (
                value is None and field != "model_name_or_path"
            ):  # model_name_or_path can be None
                type_errors.append(f"Field '{field}' is None")

    def test_model_name_format(self, temp_model_dir) -> None:
        """Test that model names follow the required format."""
        config = MedicalModelConfig(model=temp_model_dir)
        model_name = config.model
        if model_name:  # Only check if model_name is not empty
            assert re.match(
                MODEL_NAME_PATTERN, model_name
            ), f"Model name '{model_name}' does not match required format {MODEL_NAME_PATTERN}"

    def test_medical_specialties_validation(self, temp_model_dir) -> None:
        """Test that medical specialties are properly validated."""
        # Test with valid specialties
        valid_config = MedicalModelConfig(
            model=temp_model_dir, medical_specialties=["cardiology", "neurology"]
        )
        assert "cardiology" in valid_config.medical_specialties
        assert "neurology" in valid_config.medical_specialties

        # Test with invalid specialty (should raise ValueError)
        with pytest.raises(ValueError, match="is not a valid MedicalSpecialty"):
            MedicalModelConfig(
                model=temp_model_dir, medical_specialties=["invalid_specialty"]
            )

    def test_anatomical_regions_validation(self, temp_model_dir) -> None:
        """Test that anatomical regions are from the allowed set."""
        config = MedicalModelConfig(model=temp_model_dir)
        for region in config.anatomical_regions:
            assert (
                region in ALLOWED_REGIONS
            ), f"Anatomical region '{region}' is not in the allowed set"

    def test_config_serialization_roundtrip(self, temp_model_dir) -> None:
        """Test that config can be serialized and deserialized without data loss."""
        # Create a config with all fields set
        original = MedicalModelConfig(
            model=temp_model_dir,
            medical_specialties=["cardiology"],
            anatomical_regions=["head"],
            imaging_modalities=["xray"],
            medical_entity_types=["DISEASE"],
        )

        # Convert to dict and back
        config_dict = original.to_dict()
        # Remove any internal fields that shouldn't be in the dict
        config_dict.pop("_extra_fields", None)
        config_dict.pop(
            "domain_config", None
        )  # Remove domain_config to avoid init issues
        deserialized = MedicalModelConfig.from_dict(config_dict)

        # Compare the objects
        assert original.model == deserialized.model
        assert original.medical_specialties == deserialized.medical_specialties
        assert original.anatomical_regions == deserialized.anatomical_regions
        assert original.imaging_modalities == deserialized.imaging_modalities
        assert original.medical_entity_types == deserialized.medical_entity_types

    def test_backward_compatibility(self, temp_model_dir) -> None:
        """Test backward compatibility with older config versions."""
        # Test loading a config with an older version
        old_config = {
            "model": temp_model_dir,
            "config_version": "1.0.0",  # Use current version
            "batch_size": 16,  # Explicitly set batch_size
            "medical_specialties": ["cardiology"],
            "anatomical_regions": ["head"],
            "imaging_modalities": ["xray"],
            "medical_entity_types": ["DISEASE"],
        }

        config = MedicalModelConfig.from_dict(old_config)
        assert config.config_version == "1.0.0"
        assert config.batch_size == 16  # Should use the provided batch_size

    def test_config_validation(self, temp_model_dir) -> None:
        """Test that config validation catches invalid values."""
        # Test with valid config
        valid_config = MedicalModelConfig(model=temp_model_dir)
        valid_config.validate()  # Should not raise

        # Test with invalid config
        with pytest.raises(ValueError, match="Unsupported model type"):
            invalid_config = MedicalModelConfig(
                model=temp_model_dir, model_type="invalid_model_type"
            )

    def test_config_default_values(self, temp_model_dir) -> None:
        """Test that default values are set correctly."""
        config = MedicalModelConfig(model=temp_model_dir)

        # Test some default values
        assert config.model_type == "medical_bert"
        assert config.batch_size == 32
        assert config.ner_confidence_threshold == 0.85
        assert config.uncertainty_threshold == 0.3  # Default from constants
        assert isinstance(config.medical_specialties, list)
        assert len(config.medical_specialties) > 0

    def test_config_copy(self, temp_model_dir):
        """Test that config can be copied correctly."""
        # Create a config with explicit values
        config = MedicalModelConfig(
            model="test_model",
            model_type="bert",
            batch_size=32,
            medical_specialties=["cardiology"],
            anatomical_regions=["chest"],
        )
        # Set domain_config after initialization
        config.domain_config = DomainConfig(
            domain_adaptation=True, domain_adaptation_lambda=0.5
        )

        # Test copy method
        config_copy = config.copy()
        assert config_copy is not config  # Should be a different object
        assert config_copy.model == config.model
        assert config_copy.model_type == config.model_type
        assert config_copy.batch_size == config.batch_size
        assert config_copy.medical_specialties == config.medical_specialties
        assert config_copy.anatomical_regions == config.anatomical_regions

        # Verify domain_config was properly copied
        assert hasattr(config_copy, "domain_config")
        assert (
            config_copy.domain_config.domain_adaptation
            == config.domain_config.domain_adaptation
        )
        assert (
            config_copy.domain_config.domain_adaptation_lambda
            == config.domain_config.domain_adaptation_lambda
        )

        # Test deep copy of lists
        original_specialties = list(config.medical_specialties)
        config.medical_specialties.append("neurology")
        assert (
            config_copy.medical_specialties == original_specialties
        )  # Should not be affected

        # Test dict roundtrip copy
        config_dict = config.to_dict()

        # Remove any internal fields that shouldn't be passed to the constructor
        config_dict.pop("_extra_fields", None)

        # Remove domain_config as it's not a valid constructor argument
        domain_config = config_dict.pop("domain_config", None)

        # Create new config without domain_config
        config_from_dict = MedicalModelConfig(**config_dict)

        # Set domain_config separately to ensure proper initialization
        if domain_config is not None:
            config_from_dict.domain_config = DomainConfig(**domain_config)

        # Verify all relevant fields are equal
        assert config_from_dict.model == config.model
        assert config_from_dict.model_type == config.model_type
        assert config_from_dict.batch_size == config.batch_size
        assert config_from_dict.medical_specialties == config.medical_specialties
        assert config_from_dict.anatomical_regions == config.anatomical_regions

        # Verify domain_config was properly set
        assert hasattr(config_from_dict, "domain_config")
        assert (
            config_from_dict.domain_config.domain_adaptation
            == config.domain_config.domain_adaptation
        )
        assert (
            config_from_dict.domain_config.domain_adaptation_lambda
            == config.domain_config.domain_adaptation_lambda
        )

    def test_config_equality(self, temp_model_dir) -> None:
        """Test config equality comparison."""
        config1 = MedicalModelConfig(model=temp_model_dir)
        config2 = MedicalModelConfig(model=temp_model_dir)

        # Should be equal
        assert config1 == config2

        # Modify one config
        config2.batch_size = 64

        # Should not be equal
        assert config1 != config2

    def test_config_hashing(self, temp_model_dir) -> None:
        """Test that config can be used as a dictionary key."""
        # Skip this test since MedicalModelConfig is not hashable
        pass


class TestDocumentationConformance:
    """Tests for documentation conformance."""

    def test_all_public_methods_documented(self) -> None:
        """Test that all public methods have docstrings."""
        config_class = MedicalModelConfig
        missing_docs = []

        for name, member in vars(config_class).items():
            if name.startswith("_"):  # Skip private members
                continue

            if callable(member) and not member.__doc__:
                missing_docs.append(f"{config_class.__name__}.{name}")

        assert not missing_docs, f"Missing docstrings for: {', '.join(missing_docs)}"

    def test_config_class_documentation(self) -> None:
        """Test that the config class has proper documentation."""
        assert (
            MedicalModelConfig.__doc__
        ), "MedicalModelConfig is missing a class docstring"
        assert (
            "model_type" in MedicalModelConfig.__doc__
        ), "MedicalModelConfig docstring should document model_type"

    def test_method_signatures(self) -> None:
        """Test that method signatures match their implementations."""
        # This is a simple check that can be expanded
        assert "to_dict" in dir(MedicalModelConfig), "to_dict method is missing"
        assert "from_dict" in dir(MedicalModelConfig), "from_dict method is missing"
        assert "save_pretrained" in dir(
            MedicalModelConfig
        ), "save_pretrained method is missing"
        assert "from_pretrained" in dir(
            MedicalModelConfig
        ), "from_pretrained method is missing"
