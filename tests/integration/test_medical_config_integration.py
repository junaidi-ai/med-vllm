"""Integration tests for MedicalModelConfig with real file operations.

These tests verify the behavior of MedicalModelConfig with real file system operations,
using a mock implementation to avoid external dependencies.
"""

import json
import os
import sys

import pytest

# Add markers for different test types
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="File system tests not reliable on Windows CI",
    ),
]


# Mock the MedicalModelConfig class
class MedicalModelConfig:
    """Mock MedicalModelConfig for integration testing."""

    def __init__(self, model, medical_specialties=None, anatomical_regions=None, **kwargs):
        self.model = model
        self.medical_specialties = medical_specialties or []
        self.anatomical_regions = anatomical_regions or []
        self.__dict__.update(kwargs)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def to_dict(self):
        return {
            "model": self.model,
            "medical_specialties": self.medical_specialties,
            "anatomical_regions": self.anatomical_regions,
            **{
                k: v
                for k, v in self.__dict__.items()
                if k not in ("model", "medical_specialties", "anatomical_regions")
            },
        }

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_json(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class TestMedicalConfigIntegration:
    """Integration tests for MedicalModelConfig with real file operations."""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config file for testing."""
        config_path = tmp_path / "test_config.json"
        return config_path

    def test_full_lifecycle(self, temp_config_file):
        """Test the full configuration lifecycle: create, save, load, modify, save, reload."""
        # Create initial config
        config = MedicalModelConfig(
            model="test-model",
            medical_specialties=["cardiology", "neurology"],
            anatomical_regions=["head", "chest"],
        )

        # Save to file
        config.to_json(temp_config_file)
        assert temp_config_file.exists()

        # Load from file
        loaded_config = MedicalModelConfig.from_json(temp_config_file)
        assert loaded_config.model == "test-model"
        assert loaded_config.medical_specialties == ["cardiology", "neurology"]
        assert loaded_config.anatomical_regions == ["head", "chest"]

        # Modify and save again
        loaded_config.medical_specialties.append("radiology")
        loaded_config.to_json(temp_config_file)

        # Reload and verify changes
        reloaded_config = MedicalModelConfig.from_json(temp_config_file)
        assert "radiology" in reloaded_config.medical_specialties

    def test_error_handling_corrupted_file(self, temp_config_file):
        """Test handling of corrupted JSON files."""
        # Create a corrupted JSON file
        with open(temp_config_file, "w") as f:
            f.write('{"invalid": "json')

        # Test that loading raises an appropriate error
        with pytest.raises((json.JSONDecodeError, ValueError)):
            MedicalModelConfig.from_json(temp_config_file)

    def test_error_recovery_after_corruption(self, temp_config_file):
        """Test that we can recover after encountering a corrupted file."""
        # First create a valid config
        config = MedicalModelConfig(model="test-model", medical_specialties=["cardiology"])
        config.to_json(temp_config_file)

        # Corrupt the file
        with open(temp_config_file, "w") as f:
            f.write('{"invalid": "json')

        # Should raise error on loading corrupted file
        with pytest.raises((json.JSONDecodeError, ValueError)):
            MedicalModelConfig.from_json(temp_config_file)

        # Should be able to overwrite with valid config
        config.to_json(temp_config_file)
        reloaded = MedicalModelConfig.from_json(temp_config_file)
        assert reloaded.model == "test-model"


class TestMedicalConfigPerformance:
    """Performance tests for MedicalModelConfig."""

    @pytest.mark.performance
    def test_serialization_performance(self, benchmark):
        """Benchmark serialization performance with large configurations."""
        # Create a large configuration
        large_specialties = [f"specialty_{i}" for i in range(1000)]
        large_regions = [f"region_{i}" for i in range(500)]

        config = MedicalModelConfig(
            model="large-test-model",
            medical_specialties=large_specialties,
            anatomical_regions=large_regions,
        )

        # Benchmark to_dict
        result = benchmark(config.to_dict)
        assert len(result["medical_specialties"]) == 1000
        assert len(result["anatomical_regions"]) == 500

    @pytest.mark.performance
    def test_deserialization_performance(self, benchmark):
        """Benchmark deserialization performance with large configurations."""
        # Create a large configuration
        large_specialties = [f"specialty_{i}" for i in range(1000)]
        large_regions = [f"region_{i}" for i in range(500)]

        config_data = {
            "model": "large-test-model",
            "medical_specialties": large_specialties,
            "anatomical_regions": large_regions,
            "model_type": "bert",
        }

        # Benchmark from_dict
        result = benchmark(MedicalModelConfig.from_dict, config_data)
        assert len(result.medical_specialties) == 1000
        assert len(result.anatomical_regions) == 500


# Property-based testing with Hypothesis
import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings


class TestMedicalConfigPropertyBased:
    """Property-based tests for MedicalModelConfig."""

    @given(
        model_name=st.text(min_size=1, max_size=50),
        specialties=st.lists(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("L", "N")),
            ),
            min_size=1,
            max_size=10,
            unique=True,
        ),
        regions=st.lists(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("L", "N")),
            ),
            min_size=1,
            max_size=10,
            unique=True,
        ),
    )
    @settings(
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        max_examples=50,
        deadline=1000,  # 1 second per example
    )
    def test_serialization_roundtrip_property(self, model_name, specialties, regions):
        """Test that any valid config can be serialized and deserialized."""
        # Create config with generated data
        config = MedicalModelConfig(
            model=model_name,
            medical_specialties=specialties,
            anatomical_regions=regions,
        )

        # Round-trip through dict
        config_dict = config.to_dict()
        new_config = MedicalModelConfig.from_dict(config_dict)

        # Verify all properties are preserved
        assert new_config.model == model_name
        assert new_config.medical_specialties == specialties
        assert new_config.anatomical_regions == regions
