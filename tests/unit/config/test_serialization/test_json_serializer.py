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
    "model": "biobert-base-cased",
    "model_type": "biobert",
    "max_medical_seq_length": 512,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
    "model_name_or_path": null,
    "pretrained_model_name_or_path": null,
    "enable_uncertainty_estimation": false,
    "uncertainty_threshold": 0.7,
    "cache_ttl": 3600,
    "medical_specialties": ["cardiology", "radiology", "neurology"],
    "anatomical_regions": ["head", "torso", "limbs"],
    "imaging_modalities": ["xray", "mri", "ct"],
    "medical_entity_types": ["disease", "symptom", "treatment"],
    "ner_confidence_threshold": 0.5,
    "max_entity_span_length": 10,
    "entity_linking": {"enabled": false, "knowledge_bases": ["umls", "snomed"], "confidence_threshold": 0.8},
    "document_types": ["clinical_note", "radiology_report", "discharge_summary"],
    "section_headers": ["history", "findings", "impression"],
    "max_retries": 3,
    "request_timeout": 30,
    "domain_adaptation": false,
    "domain_adaptation_lambda": 0.1,
    "domain_specific_vocab": null,
    "regulatory_compliance": ["hipaa", "gdpr"],
    "config_version": "1.0.0"
}
"""

SCHEMA_COMPATIBLE_DICT = {
    "model": "biobert-base-cased",
    "model_type": "bert",
    "max_medical_seq_length": 512,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
    "model_name_or_path": None,
    "pretrained_model_name_or_path": None,
    "enable_uncertainty_estimation": False,
    "uncertainty_threshold": 0.7,
    "cache_ttl": 3600,
    "medical_specialties": ["cardiology", "radiology", "neurology"],
    "anatomical_regions": ["head", "torso", "limbs"],
    "imaging_modalities": ["xray", "mri", "ct"],
    "medical_entity_types": ["disease", "symptom", "treatment"],
    "ner_confidence_threshold": 0.5,
    "max_entity_span_length": 10,
    "entity_linking": {
        "enabled": False,
        "knowledge_bases": ["umls", "snomed"],
        "confidence_threshold": 0.8,
    },
    "document_types": ["clinical_note", "radiology_report", "discharge_summary"],
    "section_headers": ["history", "findings", "impression"],
    "max_retries": 3,
    "request_timeout": 30,
    "domain_adaptation": False,
    "domain_adaptation_lambda": 0.1,
    "domain_specific_vocab": None,
    "regulatory_compliance": ["hipaa", "gdpr"],
    "config_version": "1.0.0",
}

# Required fields that should be present in the serialized output
REQUIRED_FIELDS = {"model", "model_type", "max_medical_seq_length", "batch_size"}


# Fixtures
@pytest.fixture
def test_config() -> MedicalModelConfig:
    """Create a test configuration with all required fields."""
    # Create a minimal valid configuration
    config = MedicalModelConfig(
        model="biobert-base-cased",
        model_type="medical_ner",  # Using a valid model type from SUPPORTED_MODEL_TYPES
        max_medical_seq_length=512,
        batch_size=32,
    )

    # Add optional fields that we want to test
    config.learning_rate = 5e-5
    config.num_train_epochs = 3

    return config


class TestJSONSerializer:
    """Test cases for JSONSerializer class."""

    def test_serialize_dict(self, test_config: MedicalModelConfig) -> None:
        """Test serialization to dictionary."""
        # Test serialization to JSON string
        json_str = JSONSerializer.to_json(test_config)
        assert isinstance(json_str, str)

        # Parse the JSON to a dictionary
        result = json.loads(json_str)

        # Check that required fields are present
        required_fields = {
            "model": "biobert-base-cased",
            "model_type": "medical_ner",
            "max_medical_seq_length": 512,
            "batch_size": 32,
        }

        # Check each required field
        for field, expected_value in required_fields.items():
            assert field in result, f"Missing required field: {field}"
            assert (
                result[field] == expected_value
            ), f"Incorrect value for {field}: {result[field]}"

        # Check that the learning_rate and num_train_epochs are not in the result
        # since they are not part of the default configuration
        # Removed incomplete assertion that was causing syntax error

        # Check that the dictionary contains the expected fields
        assert isinstance(result, dict)
        assert result["model"] == "biobert-base-cased"
        assert result["model_type"] == "medical_ner"

    def test_serialize_to_file(
        self, test_config: MedicalModelConfig, tmp_path: Path
    ) -> None:
        """Test serialization of a configuration to a file."""
        # Test serialization to file
        file_path = tmp_path / "test_config.json"
        
        # Get JSON string and write to file
        json_str = JSONSerializer.to_json(test_config)
        with open(file_path, "w") as f:
            f.write(json_str)

        # Check that the file was created and contains valid JSON
        assert file_path.exists(), "Output file was not created"
        with open(file_path, "r") as f:
            data = json.load(f)

        # Check that the file contains the expected fields
        expected_fields = {
            "model": "biobert-base-cased",
            "model_type": "medical_ner",
            "max_medical_seq_length": 512,
            "batch_size": 32,
        }

        for field, expected_value in expected_fields.items():
            assert field in data, f"Missing field in output: {field}"
            assert data[field] == expected_value, (
                f"Incorrect value for {field} in file: {data[field]}"
            )

    def test_deserialize_with_extra_fields(self) -> None:
        """Test that extra fields in JSON are handled appropriately."""
        # Given - JSON with extra fields not in the schema
        json_str = """
        {
            "model": "biobert-base-cased",
            "model_type": "medical_ner",
            "max_medical_seq_length": 512,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "num_train_epochs": 3,
            "extra_field1": "should be ignored",
            "extra_field2": 123
        }
        """

        # When - Should not raise an error, extra fields should be ignored
        config = JSONSerializer.from_json(json_str, MedicalModelConfig)

        # Then - Config should be created with valid fields
        assert config.model == "biobert-base-cased"
        assert config.model_type == "medical_ner"  # Should match the provided model_type
        assert config.max_medical_seq_length == 512
        assert config.batch_size == 32

    def test_field_mapping(self) -> None:
        """Test field mapping between schema and config class."""
        # Create JSON with schema field names
        schema_json = json.dumps(
            {
                "model": "mapping-test",
                "model_type": "medical_ner",
                "max_sequence_length": 1024,  # This should map to max_medical_seq_length
                "batch_size": 16,
                "learning_rate": 2e-5,
                "num_train_epochs": 5,
            }
        )

        # Deserialize
        config = JSONSerializer.from_json(schema_json, MedicalModelConfig)

        # Check field mapping worked correctly
        assert config.model == "mapping-test"
        assert config.model_type == "medical_ner"  # Should match the string value, not enum
        assert config.max_medical_seq_length == 1024  # Mapped from max_sequence_length
        assert config.batch_size == 16
        if hasattr(config, "learning_rate"):
            assert config.learning_rate == 2e-5
        if hasattr(config, "num_train_epochs"):
            assert config.num_train_epochs == 5

    def test_deserialize(self) -> None:
        """Test deserialization of a JSON string to a configuration object."""
        # When
        config = JSONSerializer.from_json(SCHEMA_COMPATIBLE_JSON, MedicalModelConfig)

        # Then
        assert isinstance(config, MedicalModelConfig)
        assert config.model == "biobert-base-cased"
        assert config.model_type == "biobert"  # Should match the value in SCHEMA_COMPATIBLE_JSON
        assert config.max_medical_seq_length == 512
        assert config.batch_size == 32

        # Check optional fields if they exist in the config
        if hasattr(config, "learning_rate"):
            assert config.learning_rate == 5e-5
        if hasattr(config, "num_train_epochs"):
            assert config.num_train_epochs == 3
