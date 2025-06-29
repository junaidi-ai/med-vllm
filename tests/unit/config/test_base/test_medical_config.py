"""
Tests for the base medical configuration classes.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Import the actual implementation
from medvllm.medical.config.base import BaseMedicalConfig


class TestBaseMedicalConfig:
    """Test cases for BaseMedicalConfig class."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for testing model paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def config(self, temp_model_dir) -> BaseMedicalConfig:
        """Return a BaseMedicalConfig instance for testing."""
        # Create a dummy config.json file in the temp dir
        with open(os.path.join(temp_model_dir, "config.json"), "w") as f:
            f.write('{"max_position_embeddings": 4096}')
        return BaseMedicalConfig(model=temp_model_dir)

    def test_initialization(self, config: BaseMedicalConfig) -> None:
        """Test basic initialization."""
        assert config is not None
        assert hasattr(config, "model_type")
        assert hasattr(config, "to_dict")
        assert hasattr(config, "from_dict")

    def test_to_dict(self, config: BaseMedicalConfig) -> None:
        """Test conversion to dictionary."""
        # Given
        config.model_type = "test-model"
        config.custom_field = "value"

        # When
        result = config.to_dict()

        # Then
        assert isinstance(result, dict)
        assert result["model_type"] == "test-model"
        assert result["custom_field"] == "value"

    def test_from_dict(self, temp_model_dir) -> None:
        """Test creation from dictionary."""
        # Given
        # Create a dummy config.json file in the temp dir
        with open(os.path.join(temp_model_dir, "config.json"), "w") as f:
            f.write('{"max_position_embeddings": 4096}')

        data = {
            "model_type": "test-model",
            "model": temp_model_dir,
            "custom_field": "value",
        }

        # When
        config = BaseMedicalConfig.from_dict(data)

        # Then
        assert config.model_type == "test-model"
        assert config.model == temp_model_dir
        assert getattr(config, "custom_field") == "value"

    def test_validate(self, config: BaseMedicalConfig) -> None:
        """Test validation of configuration."""
        # Should not raise by default
        config.validate()

        # Test with custom validation
        config._validate_custom = MagicMock()
        config.validate()
        config._validate_custom.assert_called_once()

    def test_validate_raises(self, config: BaseMedicalConfig) -> None:
        """Test validation errors are properly raised."""

        # Given
        def failing_validation():
            raise ValueError("Validation failed")

        config._validate_custom = failing_validation

        # When/Then
        with pytest.raises(ValueError, match="Validation failed"):
            config.validate()

    def test_copy(self, config: BaseMedicalConfig) -> None:
        """Test creating a copy of the config."""
        # Given
        config.model_type = "test-model"
        config.custom_field = "value"

        # When
        copy = config.copy()

        # Then
        assert copy is not config
        assert copy.model_type == config.model_type
        assert copy.custom_field == config.custom_field

        # Modifying copy shouldn't affect original
        copy.custom_field = "new-value"
        assert config.custom_field == "value"
