"""Integration tests for the ModelRegistry with actual models."""

import os
import tempfile
import threading
import time
from typing import List

import pytest
import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from medvllm.engine.model_runner.registry import (
    ModelLoadingError,
    ModelNotFoundError,
    ModelRegistry,
    ModelType,
    get_registry,
)

# Small test models that are quick to load
TEST_MODELS = [
    "hf-internal-testing/tiny-random-bert",
    "hf-internal-testing/tiny-random-gpt2",
]


def test_registry_singleton():
    """Test that the registry follows the singleton pattern."""
    registry1 = get_registry()
    registry2 = get_registry()
    assert registry1 is registry2
    assert isinstance(registry1, ModelRegistry)


def test_register_and_load_model():
    """Test registering and loading a model."""
    registry = get_registry()
    model_name = TEST_MODELS[0]

    # Register the model
    registry.register(
        name=model_name,
        model_type=ModelType.GENERIC,
        model_class=AutoModel,
        config_class=AutoConfig,
        description="Test model",
        tags=["test", "tiny"],
        trust_remote_code=True,
    )

    # Load the model
    model = registry.load_model(model_name)
    assert model is not None
    assert isinstance(model, PreTrainedModel)

    # Verify metadata
    metadata = registry.get_metadata(model_name)
    assert metadata.name == model_name
    assert metadata.model_type == ModelType.GENERIC
    assert "test" in metadata.tags


def test_load_unregistered_model():
    """Test loading an unregistered but valid model."""
    registry = get_registry()
    model_name = TEST_MODELS[1]

    # This should work even though we didn't explicitly register it
    model = registry.load_model(model_name, trust_remote_code=True)
    assert model is not None
    assert isinstance(model, PreTrainedModel)


def test_model_caching():
    """Test that models are properly cached."""
    registry = get_registry()
    model_name = TEST_MODELS[0]

    # First load - should load from source
    model1 = registry.load_model(model_name, trust_remote_code=True)

    # Second load - should return cached instance
    model2 = registry.load_model(model_name, trust_remote_code=True)

    assert model1 is model2  # Should be the same object


def test_clear_cache():
    """Test clearing the model cache."""
    registry = get_registry()
    model_name = TEST_MODELS[0]

    # Load and cache the model
    model1 = registry.load_model(model_name, trust_remote_code=True)

    # Clear the cache
    registry.clear_cache()

    # Load again - should be a new instance
    model2 = registry.load_model(model_name, trust_remote_code=True)

    assert model1 is not model2  # Should be different objects


def test_concurrent_access():
    """Test that the registry works correctly with concurrent access."""
    registry = get_registry()
    model_name = TEST_MODELS[0]
    results = []

    def load_model():
        try:
            model = registry.load_model(model_name, trust_remote_code=True)
            results.append(model is not None)
        except Exception as e:
            results.append(False)

    # Create multiple threads to access the registry concurrently
    threads = []
    for _ in range(5):
        t = threading.Thread(target=load_model)
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Verify all threads successfully loaded the model
    assert all(results)


def test_invalid_model():
    """Test loading an invalid model."""
    registry = get_registry()
    invalid_name = "this-is-not-a-real-model"

    with pytest.raises(ModelLoadingError):
        registry.load_model(invalid_name)


def test_unregister_model():
    """Test unregistering a model."""
    registry = get_registry()
    model_name = "test-unregister-model"

    # Register a test model
    registry.register(
        name=model_name,
        model_type=ModelType.GENERIC,
        model_class=AutoModel,
        config_class=AutoConfig,
    )

    # Load to ensure it's cached
    registry.load_model(model_name, trust_remote_code=True)

    # Unregister
    registry.unregister(model_name)

    # Verify it's no longer registered
    with pytest.raises(ModelNotFoundError):
        registry.get_metadata(model_name)

    # Verify it's not in the cache
    assert model_name not in registry._model_cache  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__])
