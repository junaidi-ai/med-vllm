"""
Tests for the base ConfigSerializer class.

This module contains unit tests for the base ConfigSerializer implementation,
covering common functionality used by all serializers.
"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Type
from unittest.mock import MagicMock, patch

import pytest

# Import the actual implementation
from medvllm.medical.config.serialization.config_serializer import ConfigSerializer


class TestConfigSerializer:
    """Test cases for the base ConfigSerializer class."""
    
    class ConcreteSerializer(ConfigSerializer):
        """Concrete implementation of ConfigSerializer for testing."""
        
        @classmethod
        def _serialize_to_str(cls, data: Any, **kwargs) -> str:
            """Serialize data to a string (mock implementation)."""
            return "serialized_data"
            
        @classmethod
        def _deserialize_from_str(cls, data: str, **kwargs) -> Any:
            """Deserialize data from a string (mock implementation)."""
            return {"deserialized": "data"}
    
    @pytest.fixture
    def serializer(self) -> ConfigSerializer:
        """Return a concrete ConfigSerializer instance for testing."""
        return self.ConcreteSerializer()
    
    def test_serialize_to_file(self, serializer: ConfigSerializer) -> None:
        """Test serialization to a file."""
        with NamedTemporaryFile(suffix=".tmp", delete=False) as tmp_file:
            try:
                # When
                serializer.serialize({"key": "value"}, tmp_file.name)
                
                # Then
                with open(tmp_file.name, "r", encoding="utf-8") as f:
                    content = f.read()
                assert content == "serialized_data"
            finally:
                # Cleanup
                Path(tmp_file.name).unlink(missing_ok=True)
    
    def test_deserialize_from_file(self, serializer: ConfigSerializer) -> None:
        """Test deserialization from a file."""
        with NamedTemporaryFile(suffix=".tmp", mode="w", delete=False) as tmp_file:
            try:
                # Given
                tmp_file.write("file_content")
                tmp_file.flush()
                
                # When
                result = serializer.deserialize(tmp_file.name)
                
                # Then
                assert result == {"deserialized": "data"}
            finally:
                # Cleanup
                Path(tmp_file.name).unlink(missing_ok=True)
    
    def test_serialize_invalid_input(self, serializer: ConfigSerializer) -> None:
        """Test serialization with invalid input."""
        with pytest.raises(TypeError):
            serializer.serialize("not a dict or object with to_dict")
    
    def test_serialize_object_with_to_dict(self, serializer: ConfigSerializer) -> None:
        """Test serialization of an object with a to_dict method."""
        # Given
        class TestObject:
            def to_dict(self) -> Dict[str, Any]:
                return {"key": "value"}
                
        # When/Then - Should not raise
        result = serializer.serialize(TestObject())
        assert result == "serialized_data"
    
    def test_deserialize_invalid_type(self, serializer: ConfigSerializer) -> None:
        """Test deserialization with invalid input type."""
        with pytest.raises(TypeError):
            serializer.deserialize(123)  # type: ignore
    
    @patch("builtins.open")
    def test_serialize_file_error(self, mock_open: MagicMock, serializer: ConfigSerializer) -> None:
        """Test error handling when writing to file fails."""
        # Given
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.write.side_effect = IOError("Write error")
        mock_open.return_value = mock_file
        
        # When/Then
        with pytest.raises(IOError, match="Write error"):
            serializer.serialize({"key": "value"}, "/path/to/file")
    
    @patch("builtins.open")
    def test_deserialize_file_error(self, mock_open: MagicMock, serializer: ConfigSerializer) -> None:
        """Test error handling when reading from file fails."""
        # Given
        mock_open.side_effect = IOError("Read error")
        
        # When/Then
        with pytest.raises(IOError, match="Read error"):
            serializer.deserialize("/path/to/nonexistent/file")
