"""
Integration tests for the serialization module.

This module contains integration tests that verify the interaction between
various components of the serialization system.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import pytest

# Import the actual implementation
from medvllm.medical.config.serialization import (
    JSONSerializer,
    YAMLSerializer,
    load_config,
    save_config,
)

# Test data
SAMPLE_CONFIG = {
    "model": "test-model",
    "medical_specialties": ["cardiology", "radiology"],
    "anatomical_regions": ["head", "chest"],
    "max_seq_len": 1024,
    "dtype": "float16",
    "nested": {
        "key1": "value1",
        "key2": [1, 2, 3],
    },
}

# Skip YAML tests if PyYAML is not available
YAML_AVAILABLE = YAMLSerializer.is_available()


class TestSerializationIntegration:
    """Integration tests for serialization functionality."""

    def test_json_serialization_roundtrip(self) -> None:
        """Test roundtrip serialization with JSON."""
        # Given
        serializer = JSONSerializer(indent=2)

        # When
        serialized = serializer.serialize(SAMPLE_CONFIG)
        deserialized = serializer.deserialize(serialized)

        # Then
        assert deserialized == SAMPLE_CONFIG

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML is not installed")
    def test_yaml_serialization_roundtrip(self) -> None:
        """Test roundtrip serialization with YAML."""
        # Given
        serializer = YAMLSerializer()

        # When
        serialized = serializer.serialize(SAMPLE_CONFIG)
        deserialized = serializer.deserialize(serialized)

        # Then
        assert deserialized == SAMPLE_CONFIG

    def test_save_and_load_json(self) -> None:
        """Test saving and loading a config to/from a JSON file."""
        with TemporaryDirectory() as temp_dir:
            # Given
            file_path = Path(temp_dir) / "config.json"

            # When - Save
            save_config(SAMPLE_CONFIG, file_path)

            # Then - File should exist
            assert file_path.exists()

            # When - Load
            loaded = load_config(file_path)

            # Then
            assert loaded == SAMPLE_CONFIG

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML is not installed")
    def test_save_and_load_yaml(self) -> None:
        """Test saving and loading a config to/from a YAML file."""
        with TemporaryDirectory() as temp_dir:
            # Given
            file_path = Path(temp_dir) / "config.yaml"

            # When - Save
            save_config(SAMPLE_CONFIG, file_path)

            # Then - File should exist
            assert file_path.exists()

            # When - Load
            loaded = load_config(file_path)

            # Then
            assert loaded == SAMPLE_CONFIG

    def test_autodetect_serialization_format(self) -> None:
        """Test that the format is auto-detected from file extension."""
        with TemporaryDirectory() as temp_dir:
            # Test JSON
            json_path = Path(temp_dir) / "config.json"
            save_config(SAMPLE_CONFIG, json_path)
            loaded_json = load_config(json_path)
            assert loaded_json == SAMPLE_CONFIG

            # Test YAML if available
            if YAML_AVAILABLE:
                yaml_path = Path(temp_dir) / "config.yaml"
                save_config(SAMPLE_CONFIG, yaml_path)
                loaded_yaml = load_config(yaml_path)
                assert loaded_yaml == SAMPLE_CONFIG

    def test_serialization_with_validation(self) -> None:
        """Test that serialization works with validation."""

        # Define a simple schema
        class TestConfig:
            def __init__(self, name: str, value: int):
                self.name = name
                self.value = value

            def to_dict(self) -> Dict[str, Any]:
                return {"name": self.name, "value": self.value}

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> "TestConfig":
                return cls(**data)

        # Create a config object
        config = TestConfig("test", 42)

        # Test JSON serialization
        serializer = JSONSerializer()
        serialized = serializer.serialize(config)
        deserialized = serializer.deserialize(serialized, TestConfig)

        assert isinstance(deserialized, TestConfig)
        assert deserialized.name == "test"
        assert deserialized.value == 42

        # Test with validation
        def validate_config(data: Dict[str, Any]) -> TestConfig:
            if not isinstance(data.get("name"), str):
                raise ValueError("name must be a string")
            if not isinstance(data.get("value"), int):
                raise ValueError("value must be an integer")
            return TestConfig(**data)

        validated = serializer.deserialize(serialized, validate_config)
        assert isinstance(validated, TestConfig)

        # Test with schema validation
        class TestSchema:
            @classmethod
            def load(cls, data: Dict[str, Any]) -> TestConfig:
                return validate_config(data)

        validated_schema = serializer.deserialize(serialized, TestSchema)
        assert isinstance(validated_schema, TestConfig)
