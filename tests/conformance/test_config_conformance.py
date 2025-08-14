"""Conformance tests for configuration system.

This module contains tests that verify the configuration system
conforms to required standards and specifications.
"""

import re
from unittest.mock import patch, MagicMock

import pytest

# Mock transformers.AutoConfig before importing MedicalModelConfig
mock_auto_config = MagicMock()
mock_config = MagicMock()
mock_config.model_type = "bert"  # Default to bert for tests
mock_auto_config.from_pretrained.return_value = mock_config

with patch("transformers.AutoConfig", mock_auto_config):
    from medvllm.medical.config import MedicalModelConfig

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

    @patch("transformers.AutoConfig.from_pretrained", return_value=mock_config)
    def test_required_fields_present(self, mock_from_pretrained, temp_model_dir) -> None:
        """Test that all required fields are present in the config class."""
        # Mock the AutoConfig.from_pretrained to avoid loading actual model config
        mock_config.model_type = "bert"  # Use a valid model type from SUPPORTED_MODEL_TYPES

        # Create a config with a valid model type from SUPPORTED_MODEL_TYPES
        config = MedicalModelConfig(
            model=temp_model_dir,
            model_type="bert",  # This is a valid model type from SUPPORTED_MODEL_TYPES
            medical_specialties=["cardiology"],
            anatomical_regions=["head"],
            imaging_modalities=["xray"],
            medical_entity_types=["disease"],
        )

        # Convert to dict and check for required fields
        config_dict = config.to_dict()
        missing_fields = []
        for field, field_type in REQUIRED_FIELDS.items():
            if field not in config_dict and field != "model_name_or_path":
                missing_fields.append(field)
        assert not missing_fields, f"Missing required fields: {', '.join(missing_fields)}"

        # Verify that the model type is set correctly
        assert config.model_type == "bert"
        assert config.medical_specialties == ["cardiology"]
        assert config.anatomical_regions == ["head"]
        assert config.imaging_modalities == ["xray"]
        assert config.medical_entity_types == ["disease"]

    def test_field_types(self, temp_model_dir) -> None:
        """Test that all fields have the correct types."""
        config = MedicalModelConfig(model=temp_model_dir)
        type_errors = []

        for field_name, field_type in REQUIRED_FIELDS.items():
            value = getattr(config, field_name, None)
            if value is not None:  # Only check non-None values
                # Handle list/dict types specially
                if field_type in (list, dict):
                    assert isinstance(value, field_type), (
                        f"Field {field_name} has type {type(value).__name__}, "
                        f"expected {field_type.__name__}"
                    )
                elif hasattr(field_type, "__origin__"):
                    # Skip complex type checking for now
                    continue
                else:
                    assert isinstance(value, field_type), (
                        f"Field {field_name} has type {type(value).__name__}, "
                        f"expected {field_type.__name__}"
                    )
            elif field_name != "model_name_or_path":
                type_errors.append(f"Field '{field_name}' is None")
        assert not type_errors, "\n".join(type_errors)

    @patch("transformers.AutoConfig.from_pretrained", return_value=mock_config)
    def test_model_name_format(self, mock_from_pretrained, temp_model_dir) -> None:
        """Test that model names follow the required format."""
        # Create a config with a valid model type from SUPPORTED_MODEL_TYPES
        config = MedicalModelConfig(
            model=temp_model_dir,
            model_type="bert",  # Valid model type from SUPPORTED_MODEL_TYPES
            medical_specialties=["cardiology"],
            anatomical_regions=["head"],
            imaging_modalities=["xray"],
            medical_entity_types=["disease"],
        )
        model_name = config.model
        if model_name:  # Only check if model_name is not empty
            assert re.match(MODEL_NAME_PATTERN, model_name), (
                f"Model name '{model_name}' does not match required format " f"{MODEL_NAME_PATTERN}"
            )

    @patch("transformers.AutoConfig.from_pretrained", return_value=mock_config)
    def test_medical_specialties_validation(self, mock_from_pretrained, temp_model_dir) -> None:
        """Test that medical specialties are properly validated."""
        # Create a config with valid medical specialties
        config = MedicalModelConfig(
            model=temp_model_dir,
            model_type="medical_llm",  # Valid model type from SUPPORTED_MODEL_TYPES
            medical_specialties=["cardiology"],
            anatomical_regions=["head"],
            imaging_modalities=["xray"],
            medical_entity_types=["disease"],
        )
        assert config.medical_specialties == ["cardiology"]

        # Test with empty list
        config.medical_specialties = []
        assert config.medical_specialties == []

        # Test with invalid specialty (should be case-sensitive)
        with pytest.raises(ValueError, match="Invalid medical specialty"):
            MedicalModelConfig(
                model=temp_model_dir,
                model_type="medical_llm",
                medical_specialties=["Cardiology"],  # Should be lowercase
                anatomical_regions=["head"],
                imaging_modalities=["xray"],
                medical_entity_types=["disease"],
            )

        # Test with non-list input
        with pytest.raises(ValueError, match="must be a list"):
            MedicalModelConfig(
                model=temp_model_dir,
                model_type="medical_llm",
                medical_specialties="cardiology",  # Should be a list
                anatomical_regions=["head"],
                imaging_modalities=["xray"],
                medical_entity_types=["disease"],
            )

    @patch("transformers.AutoConfig.from_pretrained", return_value=mock_config)
    def test_anatomical_regions_validation(self, mock_from_pretrained, temp_model_dir) -> None:
        """Test that anatomical regions are from the allowed set."""
        # Create a config with valid anatomical regions
        config = MedicalModelConfig(
            model=temp_model_dir,
            model_type="medical_llm",  # Valid model type from SUPPORTED_MODEL_TYPES
            medical_specialties=["cardiology"],
            anatomical_regions=["head", "chest"],
            imaging_modalities=["xray"],
            medical_entity_types=["disease"],
        )

        # Verify all regions are in the allowed set
        for region in config.anatomical_regions:
            assert (
                region in ALLOWED_REGIONS
            ), f"Anatomical region '{region}' is not in the allowed set"

        # Test with an invalid region
        with pytest.raises(ValueError, match="is not a valid anatomical region"):
            MedicalModelConfig(
                model=temp_model_dir,
                model_type="medical_llm",
                medical_specialties=["cardiology"],
                anatomical_regions=["invalid_region"],
                imaging_modalities=["xray"],
                medical_entity_types=["disease"],
            )

    @patch("transformers.AutoConfig.from_pretrained")
    def test_config_equality(self, mock_from_pretrained, temp_model_dir) -> None:
        """Test config equality comparison."""
        print("\n[TEST] Starting test_config_equality...")

        # Mock the AutoConfig.from_pretrained to avoid loading actual model config
        mock_config = MagicMock()
        mock_config.model_type = "bert"  # Use a valid model type from SUPPORTED_MODEL_TYPES
        mock_from_pretrained.return_value = mock_config

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            print("\n[DEBUG] Creating first config...")
            # Create two identical configs
            config1 = MedicalModelConfig(
                model=temp_model_dir,
                model_type="bert",  # Valid model type from SUPPORTED_MODEL_TYPES
                medical_specialties=["cardiology"],
                anatomical_regions=["head"],
                imaging_modalities=["xray"],
                medical_entity_types=["disease"],
            )
            print("[DEBUG] First config created successfully")

            print("\n[DEBUG] Creating second config...")
            config2 = MedicalModelConfig(
                model=temp_model_dir,
                model_type="bert",  # Same model type
                medical_specialties=["cardiology"],
                anatomical_regions=["head"],
                imaging_modalities=["xray"],
                medical_entity_types=["disease"],
            )
            print("[DEBUG] Second config created successfully")

            # Debug: Print all attributes of both configs
            print("\n[DEBUG] Config1 attributes:")
            for attr in dir(config1):
                if not attr.startswith("__"):
                    try:
                        val = getattr(config1, attr, None)
                        print(f"  {attr}: {val} (type: {type(val)})")
                    except Exception as e:
                        print(f"  {attr}: [Error accessing: {e}]")

            # Debug: Compare attributes one by one
            print("\n[DEBUG] Comparing config1 and config2 attributes:")
            all_attrs = set(dir(config1) + dir(config2))
            for attr in sorted(all_attrs):
                if not attr.startswith("__") and not callable(getattr(config1, attr, None)):
                    try:
                        val1 = getattr(config1, attr, None)
                        val2 = getattr(config2, attr, None)

                        # Special handling for hf_config
                        if attr == "hf_config":
                            print("\n[DEBUG] Comparing hf_config:")
                            print(
                                f"  Config1 hf_config.model_type: {getattr(val1, 'model_type', None)}"
                            )
                            print(
                                f"  Config2 hf_config.model_type: {getattr(val2, 'model_type', None)}"
                            )
                            continue

                        if val1 != val2:
                            print(f"\n[DEBUG] Attribute '{attr}' differs:")
                            print(f"  Config1: {val1} (type: {type(val1)})")
                            print(f"  Config2: {val2} (type: {type(val2)})")

                            # Special handling for Path objects
                            if hasattr(val1, "__fspath__") or hasattr(val2, "__fspath__"):
                                print("  Path comparison:")
                                print(
                                    f"  Config1 path: {str(val1) if val1 is not None else 'None'}"
                                )
                                print(
                                    f"  Config2 path: {str(val2) if val2 is not None else 'None'}"
                                )
                    except Exception as e:
                        print(f"\n[DEBUG] Error comparing attribute '{attr}': {e}")

            print("\n[TEST] Testing config1 == config2...")
            # Test equality
            assert config1 == config2, "Identical configs should be equal"
            print("[TEST] Equality test passed")

            print("\n[TEST] Testing hash(config1) == hash(config2)...")
            assert hash(config1) == hash(config2), "Hashes of identical configs should match"
            print("[TEST] Hash equality test passed")

            print("\n[TEST] Testing inequality with different values...")
            # Test inequality with different values by creating a new config with updated batch_size
            config_dict = config1.to_dict()

            # Remove fields that are not valid __init__ parameters
            for field in ["domain_config", "version"]:
                config_dict.pop(field, None)

            config_dict["batch_size"] = 64

            # Create a new instance with the updated values
            print(f"[DEBUG] Creating new config with dict: {config_dict}")
            try:
                config3 = config1.__class__(**config_dict)

                # Debug: Print config3 attributes
                print("\n[DEBUG] Config3 attributes:")
                for attr in dir(config3):
                    if not attr.startswith("__") and not callable(getattr(config3, attr)):
                        try:
                            val = getattr(config3, attr, None)
                            print(f"  {attr}: {val} (type: {type(val)})")
                        except Exception as e:
                            print(f"  {attr}: [Error accessing: {e}]")

                assert config1 != config3, "Configs with different values should not be equal"
                assert hash(config1) != hash(config3), "Hashes of different configs should differ"
                print("[TEST] Inequality test with different values passed")
            except Exception as e:
                print(f"[ERROR] Failed to create config3: {e}")
                print(f"[DEBUG] config_dict keys: {list(config_dict.keys())}")
                raise

            # Test with different model types
            print("\n[TEST] Testing with different model types...")
            with patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=MagicMock(model_type="roberta"),
            ):
                config4 = MedicalModelConfig(
                    model=temp_model_dir,
                    model_type="roberta",  # Different model type
                    medical_specialties=["cardiology"],
                    anatomical_regions=["head"],
                    imaging_modalities=["xray"],
                    medical_entity_types=["disease"],
                )
                assert config1 != config4, "Configs with different model types should not be equal"
                assert hash(config1) != hash(
                    config4
                ), "Hashes of configs with different model types should differ"
                print("[TEST] Different model types test passed")

    def test_config_serialization_roundtrip(self, temp_model_dir) -> None:
        """Test config serialization/deserialization without data loss."""
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
        config_dict.pop("_extra_fields", None)
        config_dict.pop("domain_config", None)
        deserialized = MedicalModelConfig.from_dict(config_dict)

        # Compare the objects
        assert original.model == deserialized.model
        assert original.medical_specialties == deserialized.medical_specialties
        assert original.anatomical_regions == deserialized.anatomical_regions
        assert original.imaging_modalities == deserialized.imaging_modalities
        assert original.medical_entity_types == deserialized.medical_entity_types

    @patch("transformers.AutoConfig.from_pretrained")
    def test_backward_compatibility(self, mock_from_pretrained, temp_model_dir) -> None:
        """Test backward compatibility with older config versions."""
        # Mock the AutoConfig.from_pretrained to avoid loading actual model config
        mock_config = MagicMock()
        mock_config.model_type = "bert"  # Use a valid model type from SUPPORTED_MODEL_TYPES
        mock_from_pretrained.return_value = mock_config

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            # Test loading config from older versions
            for version, spec in COMPATIBILITY_MATRIX.items():
                # Create a minimal config with version-specific fields
                config_data = {
                    "model_type": "bert",  # Use a valid model type from SUPPORTED_MODEL_TYPES
                    "model_name_or_path": temp_model_dir,
                    "version": version,
                }

                # Add required fields for this version
                for field in spec["required_fields"]:
                    if field == "hidden_size":
                        config_data[field] = 768
                    elif field == "num_hidden_layers":
                        config_data[field] = 12
                    elif field == "medical_specialties":
                        config_data[field] = ["cardiology"]

                # Test loading the config
                config = MedicalModelConfig.from_dict(config_data)
                assert config.version == version
                assert config.model_type == "bert"  # Ensure model type is set correctly

                # Test that deprecated fields raise a warning
                for field in spec["deprecated_fields"]:
                    with pytest.warns(DeprecationWarning):
                        getattr(config, field)

                # Test that removed fields raise an AttributeError
                for field in spec["removed_fields"]:
                    with pytest.raises(AttributeError):
                        getattr(config, field)

    @patch("transformers.AutoConfig.from_pretrained")
    def test_config_validation(self, mock_from_pretrained, temp_model_dir) -> None:
        """Test that config validation catches invalid values."""
        # Mock the AutoConfig.from_pretrained to avoid loading actual model config
        mock_config = MagicMock()
        mock_config.model_type = "bert"  # Use a valid model type from SUPPORTED_MODEL_TYPES
        mock_from_pretrained.return_value = mock_config

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            # Test with valid config
            valid_config = MedicalModelConfig(
                model=temp_model_dir,
                model_type="bert",  # Valid model type from SUPPORTED_MODEL_TYPES
                medical_specialties=["cardiology"],
                anatomical_regions=["head"],
                imaging_modalities=["xray"],
                medical_entity_types=["disease"],
            )
            valid_config.validate()  # Should not raise

            # Test that model type is set correctly
            assert valid_config.model_type == "bert"

            # Test with invalid model type
            with pytest.raises(ValueError, match="Unsupported model type"):
                MedicalModelConfig(
                    model=temp_model_dir,
                    model_type="invalid_model_type",
                    medical_specialties=["cardiology"],
                    anatomical_regions=["head"],
                    imaging_modalities=["xray"],
                    medical_entity_types=["disease"],
                )

    @patch("transformers.AutoConfig.from_pretrained")
    def test_config_default_values(self, mock_from_pretrained, temp_model_dir) -> None:
        """Test that default values are set correctly."""
        # Mock the AutoConfig.from_pretrained to avoid loading actual model config
        mock_config = MagicMock()
        mock_config.model_type = "bert"  # Use a valid model type from SUPPORTED_MODEL_TYPES
        mock_from_pretrained.return_value = mock_config

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            # Create a config with only required fields
            config = MedicalModelConfig(
                model=temp_model_dir,
                model_type="bert",  # Valid model type from SUPPORTED_MODEL_TYPES
                medical_specialties=["cardiology"],
                anatomical_regions=["head"],
                imaging_modalities=["xray"],
                medical_entity_types=["disease"],
            )

            # Test default values
            assert config.model_type == "bert"  # Should use the provided model type
            assert config.batch_size == 32  # Default batch size
            assert config.ner_confidence_threshold == 0.85  # Default confidence threshold
            assert config.uncertainty_threshold == 0.3  # Default uncertainty threshold
            assert isinstance(config.medical_specialties, list)
            assert config.medical_specialties == ["cardiology"]  # Should use provided value
            assert isinstance(config.anatomical_regions, list)
            assert config.anatomical_regions == ["head"]  # Should use provided value
            assert isinstance(config.imaging_modalities, list)
            assert config.imaging_modalities == ["xray"]  # Should use provided value
            assert isinstance(config.medical_entity_types, list)
            assert config.medical_entity_types == ["disease"]  # Should use provided value

    @patch("transformers.AutoConfig.from_pretrained")
    def test_config_copy(self, mock_from_pretrained, temp_model_dir):
        """Test that config can be copied correctly."""
        # Mock the AutoConfig.from_pretrained to avoid loading actual model config
        mock_config = MagicMock()
        mock_config.model_type = "bert"  # Use a valid model type from SUPPORTED_MODEL_TYPES
        mock_from_pretrained.return_value = mock_config

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            # Create a config with explicit values
            config1 = MedicalModelConfig(
                model=temp_model_dir,
                model_type="bert",  # Valid model type from SUPPORTED_MODEL_TYPES
                medical_specialties=["cardiology"],
                anatomical_regions=["head"],
                imaging_modalities=["xray"],
                medical_entity_types=["disease"],
                batch_size=16,
                ner_confidence_threshold=0.9,
            )

            # Create a copy
            config2 = config1.copy()

            # Check that all attributes are equal
            assert config1.model == config2.model
            assert config1.model_type == config2.model_type
            assert config1.medical_specialties == config2.medical_specialties
            assert config1.anatomical_regions == config2.anatomical_regions
            assert config1.imaging_modalities == config2.imaging_modalities
            assert config1.medical_entity_types == config2.medical_entity_types
            assert config1.batch_size == config2.batch_size
            assert config1.ner_confidence_threshold == config2.ner_confidence_threshold

            # Verify the copy is a separate object
            assert config1 is not config2

            # Modify the copy and verify it doesn't affect the original
            config2.batch_size = 32
            assert config1.batch_size == 16  # Original should remain unchanged

            # Test deep copy of lists
            original_specialties = list(config1.medical_specialties)
            config1.medical_specialties.append("neurology")
            assert config2.medical_specialties == [
                "cardiology"
            ], "Modifying the original list should not affect the copy"
            assert config1.medical_specialties == [
                "cardiology",
                "neurology",
            ], "Original list should be modified"

            # Test with update method
            updates = {"batch_size": 64, "ner_confidence_threshold": 0.95}
            config_updated = config1.copy(update=updates)
            assert config_updated.batch_size == 64
            assert config_updated.ner_confidence_threshold == 0.95
            assert config_updated.model == config1.model  # Other fields should remain the same
            assert config_updated.medical_specialties == [
                "cardiology",
                "neurology",
            ]  # Should include the modified specialties

            # Test dict roundtrip copy
            config_dict = config1.to_dict()

            # Remove internal fields
            config_dict.pop("_extra_fields", None)

            # Remove domain_config as it's not a valid constructor argument
            domain_config = config_dict.pop("domain_config", None)

            # Create new config without domain_config
            config_from_dict = MedicalModelConfig(**config_dict)
        if domain_config is not None:
            # Skip domain_config as it's not needed for this test
            pass

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

        # Mock the AutoConfig.from_pretrained to avoid loading actual model config
        mock_config = MagicMock()
        mock_config.model_type = "bert"  # Use a valid model type from SUPPORTED_MODEL_TYPES
        mock_from_pretrained.return_value = mock_config

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            # Create configs with different values
            config1 = MedicalModelConfig(
                model=temp_model_dir,
                model_type="bert",  # Valid model type from SUPPORTED_MODEL_TYPES
                medical_specialties=["cardiology"],
                anatomical_regions=["head"],
                imaging_modalities=["xray"],
                medical_entity_types=["disease"],
            )

            # Create a second config with a different specialty
            with patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=MagicMock(model_type="bert"),
            ):
                config2 = MedicalModelConfig(
                    model=temp_model_dir,
                    model_type="bert",  # Same model type, different specialty
                    medical_specialties=["neurology"],
                    anatomical_regions=["head"],
                    imaging_modalities=["xray"],
                    medical_entity_types=["disease"],
                )

            # Test dictionary usage
            config_dict = {config1: "config1", config2: "config2"}
            assert config_dict[config1] == "config1", "Should retrieve config1 by key"
            assert config_dict[config2] == "config2", "Should retrieve config2 by key"

            # Test that copies with same values work as keys
            config1_copy = config1.copy()
            assert (
                config_dict[config1_copy] == "config1"
            ), "Copy with same values should work as key"

            # Verify that different configs have different hashes
            assert hash(config1) != hash(
                config2
            ), "Configs with different values should have different hashes"

            # Test with a config that has the same values as config1
            with patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=MagicMock(model_type="bert"),
            ):
                config1_same = MedicalModelConfig(
                    model=temp_model_dir,
                    model_type="bert",
                    medical_specialties=["cardiology"],  # Same as config1
                    anatomical_regions=["head"],
                    imaging_modalities=["xray"],
                    medical_entity_types=["disease"],
                )
                assert hash(config1) == hash(
                    config1_same
                ), "Configs with same values should have same hash"
                assert (
                    config_dict[config1_same] == "config1"
                ), "Config with same values should retrieve same value from dict"


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
        assert MedicalModelConfig.__doc__, "MedicalModelConfig is missing a class docstring"
        assert (
            "model_type" in MedicalModelConfig.__doc__
        ), "MedicalModelConfig docstring should document model_type"

    def test_method_signatures(self) -> None:
        """Test that method signatures match their implementations."""
        # This is a simple check that can be expanded
        required_methods = [
            ("to_dict", "to_dict method is missing"),
            ("from_dict", "from_dict method is missing"),
            ("save_pretrained", "save_pretrained method is missing"),
            ("from_pretrained", "from_pretrained method is missing"),
        ]
        for method, error_msg in required_methods:
            assert method in dir(MedicalModelConfig), error_msg
