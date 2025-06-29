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

# Constants for conformance testing
REQUIRED_FIELDS = {
    "model_type": str,
    "model_name_or_path": str,
    "hidden_size": int,
    "num_hidden_layers": int,
    "num_attention_heads": int,
    "medical_specialties": List[str],
    "anatomical_regions": List[str],
}

# Regular expression for valid model names
MODEL_NAME_PATTERN = r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$'

# Allowed medical specialties (example list, should be expanded)
ALLOWED_SPECIALTIES = {
    "cardiology", "radiology", "neurology", "oncology", "pediatrics",
    "dermatology", "gastroenterology", "endocrinology", "nephrology",
    "pulmonology", "rheumatology", "urology", "ophthalmology", "otolaryngology",
    "pathology", "psychiatry", "anesthesiology", "emergency_medicine", "family_medicine",
    "internal_medicine", "obstetrics_gynecology", "physical_medicine", "preventive_medicine",
    "radiation_oncology", "surgery"
}

# Allowed anatomical regions (example list, should be expanded)
ALLOWED_REGIONS = {
    "head", "neck", "chest", "abdomen", "pelvis", "back", "upper_limb", "lower_limb",
    "brain", "heart", "lungs", "liver", "kidneys", "stomach", "intestines", "bladder",
    "prostate", "ovaries", "uterus", "testes", "pancreas", "spleen", "thyroid", "adrenals"
}

# Configuration for backward compatibility testing
COMPATIBILITY_MATRIX = {
    "0.1.0": {
        "required_fields": ["model_type", "hidden_size", "num_hidden_layers"],
        "deprecated_fields": [],
        "removed_fields": []
    },
    "1.0.0": {
        "required_fields": ["model_type", "hidden_size", "num_hidden_layers", "medical_specialties"],
        "deprecated_fields": ["old_field"],
        "removed_fields": ["legacy_field"]
    },
    # Add more versions as needed
}


class TestConfigConformance:
    """Conformance tests for configuration system."""
    
    def test_required_fields_present(self) -> None:
        """Test that all required fields are present in the config class."""
        config = MedicalModelConfig()
        config_dict = config.to_dict()
        
        missing_fields = []
        for field, field_type in REQUIRED_FIELDS.items():
            if field not in config_dict:
                missing_fields.append(field)
        
        assert not missing_fields, f"Missing required fields: {', '.join(missing_fields)}"
    
    def test_field_types(self) -> None:
        """Test that all fields have the correct types."""
        config = MedicalModelConfig()
        config_dict = config.to_dict()
        
        type_errors = []
        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in config_dict:
                continue
                
            value = config_dict[field]
            if not isinstance(value, expected_type):
                type_errors.append(
                    f"Field '{field}' has type {type(value).__name__}, "
                    f"expected {expected_type.__name__}"
                )
        
        assert not type_errors, "\n".join(type_errors)
    
    def test_model_name_format(self) -> None:
        """Test that model names follow the required format."""
        config = MedicalModelConfig()
        model_name = config.model_name_or_path
        
        assert re.match(MODEL_NAME_PATTERN, model_name), \
            f"Model name '{model_name}' does not match pattern {MODEL_NAME_PATTERN}"
    
    def test_medical_specialties_validation(self) -> None:
        """Test that medical specialties are from the allowed set."""
        config = MedicalModelConfig()
        specialties = config.medical_specialties
        
        invalid_specialties = [s for s in specialties if s.lower() not in ALLOWED_SPECIALTIES]
        assert not invalid_specialties, \
            f"Invalid medical specialties found: {', '.join(invalid_specialties)}"
    
    def test_anatomical_regions_validation(self) -> None:
        """Test that anatomical regions are from the allowed set."""
        config = MedicalModelConfig()
        regions = config.anatomical_regions
        
        invalid_regions = [r for r in regions if r.lower() not in ALLOWED_REGIONS]
        assert not invalid_regions, \
            f"Invalid anatomical regions found: {', '.join(invalid_regions)}"
    
    def test_config_serialization_roundtrip(self) -> None:
        """Test that config can be serialized and deserialized without data loss."""
        config = MedicalModelConfig()
        config_dict = config.to_dict()
        
        # Serialize to JSON and back
        json_str = json.dumps(config_dict)
        loaded_dict = json.loads(json_str)
        
        # Create new config from loaded dict
        new_config = MedicalModelConfig.from_dict(loaded_dict)
        new_dict = new_config.to_dict()
        
        # Compare original and new dicts
        assert config_dict == new_dict, "Config changed after serialization roundtrip"
    
    def test_backward_compatibility(self) -> None:
        """Test that config maintains backward compatibility with previous versions."""
        current_version = MedicalModelConfig().config_version
        
        for version, spec in COMPATIBILITY_MATRIX.items():
            if version >= current_version:
                continue
                
            # Test that required fields exist in current version
            for field in spec["required_fields"]:
                assert hasattr(MedicalModelConfig, field), \
                    f"Required field '{field}' from version {version} is missing"
            
            # Test that deprecated fields trigger warnings
            for field in spec["deprecated_fields"]:
                if hasattr(MedicalModelConfig, field):
                    # Should trigger a deprecation warning
                    with pytest.deprecated_call():
                        getattr(MedicalModelConfig(), field)
            
            # Test that removed fields are actually removed
            for field in spec["removed_fields"]:
                assert not hasattr(MedicalModelConfig, field), \
                    f"Removed field '{field}' from version {version} still exists"
    
    def test_config_validation(self) -> None:
        """Test that config validation catches invalid values."""
        # Test with valid config (should not raise)
        config = MedicalModelConfig()
        config.validate()
        
        # Test with invalid config
        config.hidden_size = -1  # Invalid value
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = MedicalModelConfig()
        
        # Test some default values
        assert config.hidden_size > 0
        assert config.num_hidden_layers > 0
        assert config.num_attention_heads > 0
        assert config.hidden_size % config.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
    
    def test_config_copy(self) -> None:
        """Test that config can be copied correctly."""
        config = MedicalModelConfig()
        config_copy = config.copy()
        
        # Test that it's a different object
        assert config is not config_copy
        
        # Test that attributes are equal
        assert config.to_dict() == config_copy.to_dict()
        
        # Test that modifying the copy doesn't affect the original
        original_hidden_size = config.hidden_size
        config_copy.hidden_size = original_hidden_size + 1
        assert config.hidden_size == original_hidden_size
    
    def test_config_equality(self) -> None:
        """Test config equality comparison."""
        config1 = MedicalModelConfig()
        config2 = MedicalModelConfig()
        
        # Should be equal with same values
        assert config1 == config2
        
        # Should not be equal with different values
        config2.hidden_size += 1
        assert config1 != config2
    
    def test_config_hashing(self) -> None:
        """Test that config can be used as a dictionary key."""
        config1 = MedicalModelConfig()
        config2 = MedicalModelConfig()
        
        # Should be hashable
        assert isinstance(hash(config1), int)
        
        # Equal configs should have the same hash
        assert hash(config1) == hash(config2)
        
        # Different configs should (ideally) have different hashes
        config2.hidden_size += 1
        assert hash(config1) != hash(config2)


class TestDocumentationConformance:
    """Tests for documentation conformance."""
    
    def test_all_public_methods_documented(self) -> None:
        """Test that all public methods have docstrings."""
        config_class = MedicalModelConfig
        missing_docs = []
        
        for name, member in vars(config_class).items():
            if name.startswith('_'):  # Skip private members
                continue
                
            if callable(member) and not member.__doc__:
                missing_docs.append(f"{config_class.__name__}.{name}")
        
        assert not missing_docs, f"Missing docstrings for: {', '.join(missing_docs)}"
    
    def test_config_class_documentation(self) -> None:
        """Test that the config class has proper documentation."""
        assert MedicalModelConfig.__doc__, "MedicalModelConfig is missing a class docstring"
        assert "model_type" in MedicalModelConfig.__doc__, \
            "MedicalModelConfig docstring should document model_type"
    
    def test_method_signatures(self) -> None:
        """Test that method signatures match their implementations."""
        # This is a simple check that can be expanded
        assert 'to_dict' in dir(MedicalModelConfig), "to_dict method is missing"
        assert 'from_dict' in dir(MedicalModelConfig), "from_dict method is missing"
        assert 'save_pretrained' in dir(MedicalModelConfig), "save_pretrained method is missing"
        assert 'from_pretrained' in dir(MedicalModelConfig), "from_pretrained method is missing"
