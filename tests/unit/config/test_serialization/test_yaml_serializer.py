"""
Tests for YAML serialization functionality.

This module contains unit tests for the YAML serializer implementation,
covering serialization and deserialization of configuration objects.
"""

import os
import pytest
import yaml
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

from medvllm.medical.config.serialization.yaml_serializer import YAMLSerializer

# Sample configuration for testing
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

class TestYAMLSerializer:
    """Test cases for YAMLSerializer class."""

    def test_serialize_dict(self) -> None:
        """Test serialization of a dictionary to YAML string."""
        # When
        result = YAMLSerializer.to_yaml(SAMPLE_CONFIG)
        
        # Then
        assert isinstance(result, str)
        # Verify it's valid YAML by parsing it back
        parsed = YAMLSerializer.from_yaml(result)
        assert parsed == SAMPLE_CONFIG

    def test_serialize_to_file(self) -> None:
        """Test serialization of a dictionary to a YAML file."""
        with NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_file:
            try:
                # When
                YAMLSerializer.to_yaml(SAMPLE_CONFIG, tmp_file.name)
                
                # Then
                with open(tmp_file.name, "r", encoding="utf-8") as f:
                    content = YAMLSerializer.from_yaml(f.read())
                assert content == SAMPLE_CONFIG
            finally:
                # Cleanup
                Path(tmp_file.name).unlink(missing_ok=True)

    def test_deserialize(self) -> None:
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
        result = YAMLSerializer.from_yaml(yaml_str)
        
        # Then
        assert result == SAMPLE_CONFIG

    def test_deserialize_from_file(self) -> None:
        """Test deserialization from a YAML file."""
        with NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp_file:
            try:
                import yaml
                yaml.dump(SAMPLE_CONFIG, tmp_file)
                tmp_file_path = tmp_file.name
            finally:
                tmp_file.close()
                
            try:
                # When
                result = YAMLSerializer.from_yaml(tmp_file_path)
                
                # Then
                assert result == SAMPLE_CONFIG
            finally:
                # Cleanup
                Path(tmp_file_path).unlink(missing_ok=True)

    def test_serialize_with_default_flow_style(self) -> None:
        """Test serialization with default_flow_style parameter."""
        # When - with block style (default)
        result = YAMLSerializer.to_yaml(SAMPLE_CONFIG, default_flow_style=False)
        
        # Then - should be in block style
        assert "medical_specialties:" in result
        assert "- cardiology" in result
        assert "- radiology" in result
        
        # When - with flow style
        result = YAMLSerializer.to_yaml(SAMPLE_CONFIG, default_flow_style=True)
        
        # Then - should be in flow style (compact format)
        assert "medical_specialties: [cardiology, radiology]" in result or \
               "medical_specialties: [cardiology, radiology," in result

    @pytest.mark.parametrize("yaml_content,expected_type,should_raise,error_match", [
        # Valid YAML strings
        ("key: value", dict, False, None),
        ("key: [1, 2, 3, ]", dict, False, None),  # Trailing comma is valid
        ("/nonexistent/file.yaml", str, False, None),  # This is a valid YAML string
        ("- item1\n  - item2", list, False, None),   # This is valid YAML with a list
        
        # Invalid YAML strings
        ("not: a: valid: yaml", None, True, r"Invalid YAML"),
        
        # Empty content cases
        ("", None, True, "Empty YAML content"),
        ("   \n  \n  ", None, True, "Empty YAML content"),
    ])
    def test_deserialize_yaml_variations(
        self, yaml_content: str, expected_type: Any, should_raise: bool, error_match: Optional[str]
    ) -> None:
        """Test deserialization with various YAML strings."""
        if should_raise:
            with pytest.raises(expected_type, match=error_match):
                YAMLSerializer.from_yaml(yaml_content)
        else:
            result = YAMLSerializer.from_yaml(yaml_content)
            assert isinstance(result, expected_type)

    def test_deserialize_nonexistent_file(self, tmp_path: Path) -> None:
        """Test deserialization from a non-existent file."""
        non_existent_file = tmp_path / "nonexistent/file.yaml"
        # The file doesn't exist, so it should raise an error
        with pytest.raises(ValueError, match=f"File not found: .*nonexistent/file.yaml"):
            YAMLSerializer.from_yaml(non_existent_file)

    def test_serialize_file_io_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test handling of IOError during file serialization."""
        # Create a read-only directory to cause permission error
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o400)  # Read-only
        
        # Try to write to a file in the read-only directory
        with pytest.raises(ValueError, match="Failed to serialize"):
            YAMLSerializer.to_yaml(SAMPLE_CONFIG, str(read_only_dir / "test.yaml"))

    def test_deserialize_file_io_error(self, tmp_path: Path) -> None:
        """Test handling of IOError during file deserialization."""
        # Create a file but make it unreadable
        unreadable_file = tmp_path / "unreadable.yaml"
        unreadable_file.touch()
        unreadable_file.chmod(0o000)  # No permissions
        
        with pytest.raises(ValueError, match="Failed to deserialize"):
            YAMLSerializer.from_yaml(str(unreadable_file))

    def test_pyyaml_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test behavior when PyYAML is not installed."""
        # Given
        monkeypatch.setattr("medvllm.medical.config.serialization.yaml_serializer.PYYAML_AVAILABLE", False)
        
        # When/Then
        with pytest.raises(ImportError):
            YAMLSerializer.to_yaml(SAMPLE_CONFIG)
        
        with pytest.raises(ImportError):
            YAMLSerializer.from_yaml("key: value")
            
        # Cleanup - restore the original value
        monkeypatch.undo()
