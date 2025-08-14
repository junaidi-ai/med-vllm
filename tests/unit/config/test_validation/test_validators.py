"""
Tests for the validation validators.

This module contains unit tests for the validation functionality,
including the MedicalConfigValidator and other validators.
"""

from typing import Any

import pytest

from medvllm.medical.config.validation.exceptions import (
    FieldValueError,
    ValidationError,
)

# Import the actual implementation
from medvllm.medical.config.validation.validators import MedicalConfigValidator

# Remove unused imports


# Test data
SAMPLE_CONFIG = {
    "model": "test-model",
    "medical_specialties": ["cardiology", "radiology"],
    "anatomical_regions": ["head"],
    "tensor_parallel_size": 4,
    "entity_linking": {"enabled": True, "knowledge_bases": ["umls", "snomed"]},
}


class MockConfig:
    """Mock configuration object for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


class TestMedicalConfigValidator:
    """Test cases for MedicalConfigValidator class."""

    @pytest.fixture
    def validator(self) -> MedicalConfigValidator:
        """Return a MedicalConfigValidator instance for testing."""
        return MedicalConfigValidator()

    @pytest.fixture
    def config(self) -> MockConfig:
        """Return a sample configuration object for testing."""
        return MockConfig(**SAMPLE_CONFIG)

    def test_validate_tensor_parallel_size_valid(self, validator: MedicalConfigValidator) -> None:
        """Test validation of valid tensor_parallel_size values."""
        # Valid values should not raise exceptions
        for value in [1, 2, 4, 8]:
            validator.validate_tensor_parallel_size(value)

    @pytest.mark.parametrize(
        "value,expected_error",
        [
            (0, "must be between 1 and 8"),
            (9, "must be between 1 and 8"),
            (-1, "must be between 1 and 8"),
            (None, "must be between 1 and 8"),
            ("not_an_int", "must be between 1 and 8"),
        ],
    )
    def test_validate_tensor_parallel_size_invalid(
        self, validator: MedicalConfigValidator, value: Any, expected_error: str
    ) -> None:
        """Test validation of invalid tensor_parallel_size values."""
        with pytest.raises(FieldValueError, match=expected_error):
            validator.validate_tensor_parallel_size(value)

    def test_validate_entity_linking_valid(
        self, validator: MedicalConfigValidator, config: MockConfig
    ) -> None:
        """Test validation of valid entity linking configuration."""
        # Should not raise
        validator.validate_entity_linking(config)

    def test_validate_entity_linking_disabled(self, validator: MedicalConfigValidator) -> None:
        """Test validation when entity linking is disabled."""
        config = MockConfig(entity_linking={"enabled": False, "knowledge_bases": []})
        # Should not raise
        validator.validate_entity_linking(config)

    def test_validate_entity_linking_missing_knowledge_bases(
        self, validator: MedicalConfigValidator
    ) -> None:
        """Test validation when entity linking has no knowledge bases."""
        config = MockConfig(entity_linking={"enabled": True, "knowledge_bases": []})
        with pytest.raises(ValidationError, match="no knowledge bases are specified"):
            validator.validate_entity_linking(config)

    def test_validate_entity_linking_missing_entity_linking(
        self, validator: MedicalConfigValidator
    ) -> None:
        """Test validation when entity_linking attribute is missing."""
        config = MockConfig()  # No entity_linking attribute
        # Should not raise
        validator.validate_entity_linking(config)

    def test_validate_medical_parameters(
        self, validator: MedicalConfigValidator, config: MockConfig
    ) -> None:
        """Test validation of all medical parameters."""
        # Should not raise with valid config
        validator.validate_medical_parameters(config)

    def test_validate_medical_parameters_missing_attributes(
        self, validator: MedicalConfigValidator
    ) -> None:
        """Test validation when some attributes are missing."""
        config = MockConfig()  # Minimal config
        # Should not raise
        validator.validate_medical_parameters(config)

    @pytest.mark.parametrize(
        "param_name,invalid_value,expected_error",
        [
            ("tensor_parallel_size", 0, "must be between 1 and 8"),
            (
                "entity_linking",
                {"enabled": True, "knowledge_bases": []},
                "no knowledge bases",
            ),
        ],
    )
    def test_validate_medical_parameters_invalid(
        self,
        validator: MedicalConfigValidator,
        param_name: str,
        invalid_value: Any,
        expected_error: str,
    ) -> None:
        """Test validation of invalid medical parameters."""
        config = MockConfig(**{param_name: invalid_value})
        with pytest.raises(ValidationError, match=expected_error):
            validator.validate_medical_parameters(config)

    def test_warn_deprecated(self, validator: MedicalConfigValidator, recwarn) -> None:
        """Test deprecation warning helper method."""
        # When
        validator.warn_deprecated("old_param", "2.0.0", "new_param")

        # Then
        assert len(recwarn) == 1
        warning = recwarn.pop(DeprecationWarning)
        assert "old_param' is deprecated" in str(warning.message)
        assert "version 2.0.0" in str(warning.message)
        assert "Use 'new_param' instead" in str(warning.message)

    def test_warn_deprecated_no_alternative(
        self, validator: MedicalConfigValidator, recwarn
    ) -> None:
        """Test deprecation warning without alternative parameter."""
        # When
        validator.warn_deprecated("old_param", "2.0.0")

        # Then
        assert len(recwarn) == 1
        warning = recwarn.pop(DeprecationWarning)
        assert "old_param' is deprecated" in str(warning.message)
        assert "version 2.0.0" in str(warning.message)
        assert "instead" not in str(warning.message)
