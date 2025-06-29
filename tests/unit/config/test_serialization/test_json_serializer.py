"""
Tests for JSON serialization functionality.

This module contains unit tests for the JSON serializer implementation,
covering serialization and deserialization of configuration objects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pydantic import ValidationError

from medvllm.medical.config.models.medical_config import MedicalModelConfig
from medvllm.medical.config.models.schema import MedicalModelConfigSchema, ModelType
from medvllm.medical.config.serialization.json_serializer import JSONSerializer

# Test data - must match the schema exactly
SCHEMA_COMPATIBLE_JSON = """
{
    "model": "test-model",
    "model_type": "bert",
    "max_sequence_length": 512,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_train_epochs": 3
}
"""

SCHEMA_COMPATIBLE_DICT = {
    "model": "test-model",
    "model_type": "bert",
    "max_sequence_length": 512,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_train_epochs": 3
}

# Schema fields that should be present in the serialized output
SCHEMA_FIELDS = {
    "model", 
    "model_type", 
    "max_sequence_length", 
    "batch_size",
    "learning_rate",
    "num_train_epochs"
}

# Fixtures
@pytest.fixture
def test_config() -> MedicalModelConfig:
    """Create a test configuration with only schema-compatible fields."""
    # Create a dictionary with only the schema fields
    config_data = {
        "model": "test-model",
        "model_type": ModelType.BERT,
        "max_medical_seq_length": 512,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "num_train_epochs": 3
    }
    
    # Create a new instance with only these fields
    # This avoids initializing all the extra fields in MedicalModelConfig
    config = MedicalModelConfig.__new__(MedicalModelConfig)
    
    # Set only the schema fields
    for key, value in config_data.items():
        setattr(config, key, value)
    
    return config


class TestJSONSerializer:
    """Test cases for JSONSerializer class."""

    def test_serialize_dict(self, test_config: MedicalModelConfig) -> None:
        """Test serialization of a configuration to JSON string."""
        # When
        result = JSONSerializer.to_json(test_config)
        
        # Then
        assert isinstance(result, str)
        
        # Parse the JSON to verify structure
        parsed = json.loads(result)
        
        # Verify only schema fields are present
        assert set(parsed.keys()) == SCHEMA_FIELDS
        
        # Check all schema fields are present with correct values
        assert parsed["model"] == "test-model"
        assert parsed["model_type"] == "bert"
        assert parsed["max_sequence_length"] == 512
        assert parsed["batch_size"] == 32
        assert parsed["learning_rate"] == 5e-5
        assert parsed["num_train_epochs"] == 3
        
        # When deserialized back, it should match the original for schema fields
        deserialized = JSONSerializer.from_json(result, MedicalModelConfig)
        assert deserialized.model == test_config.model
        assert deserialized.model_type == test_config.model_type
        assert deserialized.max_medical_seq_length == test_config.max_medical_seq_length
        assert deserialized.batch_size == test_config.batch_size

    def test_serialize_to_file(self, tmp_path: Path, test_config: MedicalModelConfig) -> None:
        """Test serialization of a configuration to a JSON file."""
        # Given
        file_path = tmp_path / "output_config.json"
        
        # When
        JSONSerializer.to_json(test_config, file_path)
        
        # Then
        assert file_path.exists()
        with open(file_path, "r") as f:
            content = json.load(f)
        
        # Verify only schema fields are present
        assert set(content.keys()) == SCHEMA_FIELDS
        
        # Check values
        assert content["model"] == test_config.model
        assert content["model_type"] == test_config.model_type.value
        assert content["max_sequence_length"] == test_config.max_medical_seq_length
        assert content["batch_size"] == test_config.batch_size
        if hasattr(test_config, 'learning_rate'):
            assert content["learning_rate"] == test_config.learning_rate
        if hasattr(test_config, 'num_train_epochs'):
            assert content["num_train_epochs"] == test_config.num_train_epochs

    def test_deserialize_with_extra_fields(self) -> None:
        """Test that extra fields in JSON are not allowed (schema has extra='forbid')."""
        # Given - JSON with extra fields not in the schema
        json_str = """
        {
            "model": "test-model",
            "model_type": "bert",
            "max_sequence_length": 512,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "num_train_epochs": 3,
            "extra_field1": "should cause error",
            "extra_field2": 123
        }
        """
        
        # When/Then - should raise ValueError due to extra fields
        with pytest.raises(ValueError) as exc_info:
            JSONSerializer.from_json(json_str, MedicalModelConfig)
            
        # Verify the error message mentions the validation error
        error_str = str(exc_info.value)
        assert "validation error" in str(error_str).lower()
        assert "extra_field1" in str(error_str) or "extra_forbidden" in str(error_str)

    def test_field_mapping(self) -> None:
        """Test field mapping between schema and config class."""
        # Create JSON with schema field names
        schema_json = json.dumps({
            "model": "mapping-test",
            "model_type": "bert",
            "max_sequence_length": 1024,  # This should map to max_medical_seq_length
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_train_epochs": 5
        })

        # Deserialize
        config = JSONSerializer.from_json(schema_json, MedicalModelConfig)

        # Check field mapping worked correctly
        assert config.model == "mapping-test"
        assert config.model_type == ModelType.BERT
        assert config.max_medical_seq_length == 1024  # Mapped from max_sequence_length
        assert config.batch_size == 16
        if hasattr(config, 'learning_rate'):
            assert config.learning_rate == 2e-5
        if hasattr(config, 'num_train_epochs'):
            assert config.num_train_epochs == 5

    def test_deserialize(self) -> None:
        """Test deserialization of a configuration from a JSON string."""
        # Given - use schema-compatible JSON
        json_str = SCHEMA_COMPATIBLE_JSON
        
        # When
        deserialized = JSONSerializer.from_json(json_str, MedicalModelConfig)
        
        # Then
        assert deserialized.model == "test-model"
        assert deserialized.model_type == ModelType.BERT
        assert deserialized.max_medical_seq_length == 512  # Mapped from max_sequence_length
        assert deserialized.batch_size == 32
        
        # Check optional fields if they exist in the config
        if hasattr(deserialized, 'learning_rate'):
            assert deserialized.learning_rate == 5e-5
        if hasattr(deserialized, 'num_train_epochs'):
            assert deserialized.num_train_epochs == 3
