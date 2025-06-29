"""
Tests for JSON serialization functionality.

This module contains unit tests for the JSON serializer implementation,
covering serialization and deserialization of configuration objects.
"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Import the actual implementation
from medvllm.medical.config.serialization.json_serializer import JSONSerializer

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


class TestJSONSerializer:
    """Test cases for JSONSerializer class."""

    @pytest.fixture
    def serializer(self) -> JSONSerializer:
        """Return a JSONSerializer instance for testing."""
        return JSONSerializer()

    def test_serialize_dict(self, serializer: JSONSerializer) -> None:
        """Test serialization of a dictionary to JSON string."""
        # When
        result = serializer.serialize(SAMPLE_CONFIG)
        
        # Then
        assert isinstance(result, str)
        # Verify it's valid JSON by parsing it back
        parsed = json.loads(result)
        assert parsed == SAMPLE_CONFIG

    def test_serialize_to_file(self, serializer: JSONSerializer) -> None:
        """Test serialization of a dictionary to a JSON file."""
        with NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            try:
                # When
                serializer.serialize(SAMPLE_CONFIG, tmp_file.name)
                
                # Then
                with open(tmp_file.name, "r", encoding="utf-8") as f:
                    content = json.load(f)
                assert content == SAMPLE_CONFIG
            finally:
                # Cleanup
                Path(tmp_file.name).unlink(missing_ok=True)

    def test_deserialize(self, serializer: JSONSerializer) -> None:
        """Test deserialization from a JSON string."""
        # Given
        json_str = json.dumps(SAMPLE_CONFIG)
        
        # When
        result = serializer.deserialize(json_str)
        
        # Then
        assert result == SAMPLE_CONFIG

    def test_deserialize_from_file(self, serializer: JSONSerializer) -> None:
        """Test deserialization from a JSON file."""
        with NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp_file:
            try:
                # Given
                json.dump(SAMPLE_CONFIG, tmp_file)
                tmp_file.flush()
                
                # When
                result = serializer.deserialize(tmp_file.name)
                
                # Then
                assert result == SAMPLE_CONFIG
            finally:
                # Cleanup
                Path(tmp_file.name).unlink(missing_ok=True)

    def test_serialize_with_indent(self) -> None:
        """Test serialization with indentation for pretty printing."""
        # Given
        serializer = JSONSerializer(indent=2)
        
        # When
        result = serializer.serialize(SAMPLE_CONFIG)
        
        # Then
        assert '\n' in result  # Should be pretty printed with newlines
        parsed = json.loads(result)
        assert parsed == SAMPLE_CONFIG

    @pytest.mark.parametrize("invalid_input", [
        "not a json string",
        "{'invalid': 'json'}",  # JSON uses double quotes
        "[1, 2, 3,]",  # Trailing comma
    ])
    def test_deserialize_invalid_json(self, serializer: JSONSerializer, invalid_input: str) -> None:
        """Test deserialization with invalid JSON input."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            serializer.deserialize(invalid_input)

    def test_deserialize_nonexistent_file(self, serializer: JSONSerializer) -> None:
        """Test deserialization from a non-existent file."""
        with pytest.raises(FileNotFoundError):
            serializer.deserialize("/nonexistent/file.json")

    def test_serialize_invalid_object(self, serializer: JSONSerializer) -> None:
        """Test serialization of an object that's not JSON-serializable."""
        class NonSerializable:
            pass
            
        with pytest.raises(TypeError):
            serializer.serialize({"obj": NonSerializable()})

    @patch("json.dump")
    def test_serialize_file_io_error(self, mock_dump: MagicMock, serializer: JSONSerializer) -> None:
        """Test handling of IOError during file serialization."""
        mock_dump.side_effect = IOError("Failed to write to file")
        
        with pytest.raises(IOError, match="Failed to write to file"):
            with NamedTemporaryFile(suffix=".json") as tmp_file:
                serializer.serialize(SAMPLE_CONFIG, tmp_file.name)

    @patch("json.load")
    def test_deserialize_file_io_error(self, mock_load: MagicMock, serializer: JSONSerializer) -> None:
        """Test handling of IOError during file deserialization."""
        mock_load.side_effect = IOError("Failed to read file")
        
        with pytest.raises(IOError, match="Failed to read file"):
            with NamedTemporaryFile(suffix=".json") as tmp_file:
                serializer.deserialize(tmp_file.name)
