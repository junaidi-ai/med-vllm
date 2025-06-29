"""
Tests for YAML serialization functionality.

This module contains unit tests for the YAML serializer implementation,
covering serialization and deserialization of configuration objects.
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Import the actual implementation
from medvllm.medical.config.serialization.yaml_serializer import YAMLSerializer, YAMLError

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

# Skip tests if PyYAML is not available
pytestmark = pytest.mark.skipif(
    not YAMLSerializer.is_available(),
    reason="PyYAML is not installed"
)


class TestYAMLSerializer:
    """Test cases for YAMLSerializer class."""

    @pytest.fixture
    def serializer(self) -> YAMLSerializer:
        """Return a YAMLSerializer instance for testing."""
        return YAMLSerializer()

    def test_serialize_dict(self, serializer: YAMLSerializer) -> None:
        """Test serialization of a dictionary to YAML string."""
        # When
        result = serializer.serialize(SAMPLE_CONFIG)
        
        # Then
        assert isinstance(result, str)
        # Verify it's valid YAML by parsing it back
        parsed = serializer.deserialize(result)
        assert parsed == SAMPLE_CONFIG

    def test_serialize_to_file(self, serializer: YAMLSerializer) -> None:
        """Test serialization of a dictionary to a YAML file."""
        with NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_file:
            try:
                # When
                serializer.serialize(SAMPLE_CONFIG, tmp_file.name)
                
                # Then
                with open(tmp_file.name, "r", encoding="utf-8") as f:
                    content = serializer.deserialize(f.read())
                assert content == SAMPLE_CONFIG
            finally:
                # Cleanup
                Path(tmp_file.name).unlink(missing_ok=True)

    def test_deserialize(self, serializer: YAMLSerializer) -> None:
        """Test deserialization from a YAML string."""
        # Given
        yaml_str = """
        model: test-model
        medical_specialties:
          - cardiology
          - radiology
        anatomical_regions:
          - head
          - chest
        max_seq_len: 1024
        dtype: float16
        nested:
          key1: value1
          key2:
            - 1
            - 2
            - 3
        """
        
        # When
        result = serializer.deserialize(yaml_str)
        
        # Then
        assert result == SAMPLE_CONFIG

    def test_deserialize_from_file(self, serializer: YAMLSerializer) -> None:
        """Test deserialization from a YAML file."""
        with NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp_file:
            try:
                # Given
                yaml_content = """
                model: test-model
                medical_specialties: [cardiology, radiology]
                anatomical_regions: [head, chest]
                max_seq_len: 1024
                dtype: float16
                nested:
                  key1: value1
                  key2: [1, 2, 3]
                """
                tmp_file.write(yaml_content)
                tmp_file.flush()
                
                # When
                result = serializer.deserialize(tmp_file.name)
                
                # Then
                assert result == SAMPLE_CONFIG
            finally:
                # Cleanup
                Path(tmp_file.name).unlink(missing_ok=True)

    def test_serialize_with_default_flow_style(self) -> None:
        """Test serialization with default_flow_style parameter."""
        # Given
        serializer = YAMLSerializer(default_flow_style=False)
        
        # When
        result = serializer.serialize(SAMPLE_CONFIG)
        
        # Then
        # Should use block style (not flow style)
        assert '  - ' in result  # Indicates block style for lists
        parsed = serializer.deserialize(result)
        assert parsed == SAMPLE_CONFIG

    @pytest.mark.parametrize("invalid_yaml", [
        "not: a: valid: yaml",
        "- item1\n - item2",
        "key: [1, 2, 3,]",
    ])
    def test_deserialize_invalid_yaml(self, serializer: YAMLSerializer, invalid_yaml: str) -> None:
        """Test deserialization with invalid YAML input."""
        with pytest.raises(YAMLError):
            serializer.deserialize(invalid_yaml)

    def test_deserialize_nonexistent_file(self, serializer: YAMLSerializer) -> None:
        """Test deserialization from a non-existent file."""
        with pytest.raises(FileNotFoundError):
            serializer.deserialize("/nonexistent/file.yaml")

    @patch("yaml.safe_dump")
    def test_serialize_file_io_error(self, mock_dump: MagicMock, serializer: YAMLSerializer) -> None:
        """Test handling of IOError during file serialization."""
        mock_dump.side_effect = IOError("Failed to write to file")
        
        with pytest.raises(IOError, match="Failed to write to file"):
            with NamedTemporaryFile(suffix=".yaml") as tmp_file:
                serializer.serialize(SAMPLE_CONFIG, tmp_file.name)

    @patch("yaml.safe_load")
    def test_deserialize_file_io_error(self, mock_load: MagicMock, serializer: YAMLSerializer) -> None:
        """Test handling of IOError during file deserialization."""
        mock_load.side_effect = IOError("Failed to read file")
        
        with pytest.raises(IOError, match="Failed to read file"):
            with NamedTemporaryFile(suffix=".yaml") as tmp_file:
                serializer.deserialize(tmp_file.name)

    def test_pyyaml_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test behavior when PyYAML is not installed."""
        # Given
        monkeypatch.setattr("medvllm.medical.config.serialization.yaml_serializer.PYYAML_AVAILABLE", False)
        
        # When/Then
        with pytest.raises(ImportError, match="PyYAML is required for YAML serialization"):
            YAMLSerializer()
            
        # Cleanup - restore the original value
        monkeypatch.undo()
