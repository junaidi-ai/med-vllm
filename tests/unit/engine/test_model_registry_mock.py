"""Unit tests for the ModelRegistry using mocks."""

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast
from unittest.mock import ANY, MagicMock, Mock, create_autospec, patch

import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel

# Mock the transformers module
sys.modules["transformers"] = MagicMock()
sys.modules["transformers.modeling_utils"] = MagicMock()
sys.modules["transformers.configuration_utils"] = MagicMock()

# Now import our registry
from medvllm.engine.model_runner.registry import (
    ModelLoadingError,
    ModelMetadata,
    ModelNotFoundError,
    ModelRegistrationError,
    ModelRegistry,
    ModelType,
    get_registry,
)

# Type variables for generic typing
M = TypeVar("M", bound=PreTrainedModel)
C = TypeVar("C", bound=PretrainedConfig)


class MockModel(PreTrainedModel):
    def __init__(self, config: Optional[C] = None, **kwargs):
        super().__init__(config or MockConfig())

    @classmethod
    def from_pretrained(
        cls: Type[MockModelT],
        pretrained_model_name_or_path: str | os.PathLike[Any],
        *args: Any,
        **kwargs: Any,
    ) -> MockModelT:
        return cast(MockModelT, cls())


# Type alias for the mock model class
MockModelT = TypeVar("MockModelT", bound="MockModel")
MockConfigT = TypeVar("MockConfigT", bound="MockConfig")


# Create a mock config class that inherits from PretrainedConfig
class MockConfig(PretrainedConfig):
    model_type: str = "mock"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls: Type[MockConfigT],
        pretrained_model_name_or_path: str | os.PathLike[Any],
        cache_dir: str | os.PathLike[Any] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> MockConfigT:
        return cast(MockConfigT, cls())


def test_registry_singleton():
    """Test that the registry follows the singleton pattern."""
    registry1 = get_registry()
    registry2 = get_registry()
    assert registry1 is registry2
    assert isinstance(registry1, ModelRegistry)


@patch("transformers.PreTrainedModel.from_pretrained")
def test_register_and_retrieve_model(mock_from_pretrained):
    """Test registering and retrieving a model from the registry."""
    registry = get_registry()
    registry.clear()  # Clear any existing models

    # Configure the mock
    mock_model = MockModel()
    mock_from_pretrained.return_value = mock_model

    # Register a mock model
    registry.register(
        name="test-model",
        model_type=ModelType.BIOMEDICAL,
        model_class=MockModel,
        config_class=MockConfig,
        description="Test model",
        tags=["test", "mock"],
    )

    # Verify the model was registered
    metadata = registry.get_metadata("test-model")
    assert metadata is not None
    assert metadata.name == "test-model"
    assert metadata.model_type == ModelType.BIOMEDICAL
    assert "test" in metadata.tags
    assert "mock" in metadata.tags


@patch("transformers.PreTrainedModel.from_pretrained")
def test_load_model(mock_from_pretrained):
    """Test loading a model through the registry."""
    registry = get_registry()
    registry.clear()  # Clear any existing models

    # Configure the mock
    mock_model = MockModel()
    mock_from_pretrained.return_value = mock_model

    # Register the mock model
    registry.register(
        name="test-load-model",
        model_type=ModelType.BIOMEDICAL,
        model_class=MockModel,
        config_class=MockConfig,
    )

    # Load the model
    model = registry.load_model("test-load-model")

    # Verify the model was loaded correctly
    assert model is not None
    assert isinstance(model, MockModel)
    mock_from_pretrained.assert_called_once()


@patch("transformers.PreTrainedModel.from_pretrained")
def test_model_caching(mock_from_pretrained):
    """Test that models are properly cached."""
    registry = get_registry()
    registry.clear()  # Clear any existing models

    # Configure the mock
    mock_model = MockModel()
    mock_from_pretrained.return_value = mock_model

    # Register the mock model
    registry.register(
        name="test-cache-model",
        model_type=ModelType.BIOMEDICAL,
        model_class=MockModel,
        config_class=MockConfig,
    )

    # First load - should call from_pretrained
    model1 = registry.load_model("test-cache-model")
    assert mock_from_pretrained.call_count == 1

    # Reset the mock to verify it's not called again
    mock_from_pretrained.reset_mock()

    # Second load - should use cached model
    model2 = registry.load_model("test-cache-model")

    # Verify the same instance is returned and from_pretrained wasn't called again
    assert model1 is model2
    mock_from_pretrained.assert_not_called()


@patch("transformers.PreTrainedModel.from_pretrained")
def test_clear_cache(mock_from_pretrained):
    """Test clearing the model cache."""
    registry = get_registry()
    registry.clear()  # Clear any existing models

    # Configure the mock
    mock_model = MockModel()
    mock_from_pretrained.return_value = mock_model

    # Register the mock model
    registry.register(
        name="test-clear-cache",
        model_type=ModelType.BIOMEDICAL,
        model_class=MockModel,
        config_class=MockConfig,
    )

    # Load the model to cache it
    registry.load_model("test-clear-cache")

    # Verify it's in the cache
    assert "test-clear-cache" in registry._model_cache

    # Clear the cache
    registry.clear_cache()

    # The cache should be empty
    assert "test-clear-cache" not in registry._model_cache


@patch("transformers.PreTrainedModel.from_pretrained")
def test_unregister_model(mock_from_pretrained):
    """Test unregistering a model."""
    registry = get_registry()
    registry.clear()  # Clear any existing models

    # Configure the mock
    mock_model = MockModel()
    mock_from_pretrained.return_value = mock_model

    # Register a test model
    registry.register(
        name="test-unregister",
        model_type=ModelType.BIOMEDICAL,
        model_class=MockModel,
        config_class=MockConfig,
    )

    # Load to ensure it's cached
    registry.load_model("test-unregister")
    assert "test-unregister" in registry._model_cache

    # Unregister the model
    registry.unregister("test-unregister")

    # Verify it's no longer registered
    with pytest.raises(ModelNotFoundError):
        registry.get_metadata("test-unregister")

    # Verify it's not in the cache
    assert "test-unregister" not in registry._model_cache


@patch("transformers.PreTrainedModel.from_pretrained")
def test_concurrent_access(mock_from_pretrained):
    """Test that the registry works correctly with concurrent access."""
    registry = get_registry()
    registry.clear()  # Clear any existing models

    # Configure the mock
    mock_model = MockModel()
    mock_from_pretrained.return_value = mock_model

    # Register a mock model
    registry.register(
        name="test-concurrent",
        model_type=ModelType.BIOMEDICAL,
        model_class=MockModel,
        config_class=MockConfig,
    )

    results = []
    result_lock = threading.Lock()

    def load_model():
        try:
            model = registry.load_model("test-concurrent")
            with result_lock:
                results.append(model is not None and isinstance(model, MockModel))
        except Exception as e:
            with result_lock:
                results.append(False)

    # Create multiple threads that try to load the model simultaneously
    threads = [threading.Thread(target=load_model) for _ in range(10)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify all threads successfully loaded the model
    assert all(results) and len(results) == 10

    # Verify from_pretrained was only called once despite multiple threads
    assert mock_from_pretrained.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__])
