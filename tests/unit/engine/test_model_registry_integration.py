"""Integration tests for the ModelRegistry with actual models."""

import os
import tempfile
import threading
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

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
    lock = threading.Lock()

    def load_model():
        try:
            # Random delay to increase chance of race conditions
            time.sleep(random.uniform(0, 0.1))
            model = registry.load_model(model_name, trust_remote_code=True)
            with lock:
                results.append(model is not None)
        except Exception as e:
            with lock:
                results.append(False)

    # Create and start multiple threads
    num_threads = 20
    threads = [threading.Thread(target=load_model) for _ in range(num_threads)]
    for t in threads:
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Verify all threads successfully loaded the model
    assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"
    assert all(results), "Some threads failed to load the model"
    
    # Verify the model was only loaded once (check load count in metadata)
    metadata = registry.get_metadata(model_name)
    assert metadata.load_count == 1, f"Expected load_count=1, got {metadata.load_count}"


def test_invalid_model():
    """Test loading an invalid model."""
    registry = get_registry()
    
    # Test with non-existent model name
    with pytest.raises(ModelLoadingError):
        registry.load_model("this-is-not-a-real-model")
    
    # Test with invalid model class
    registry.register(
        name="invalid-model-class",
        model_type=ModelType.GENERIC,
        model_class=object,  # Not a valid model class
        config_class=AutoConfig,
    )
    with pytest.raises(ModelLoadingError):
        registry.load_model("invalid-model-class")
    
    # Test with invalid config
    registry.register(
        name="invalid-config",
        model_type=ModelType.GENERIC,
        model_class=AutoModel,
        config_class=object,  # Not a valid config class
    )
    with pytest.raises(ModelLoadingError):
        registry.load_model("invalid-config")


def test_unregister_model():
    """Test unregistering a model."""
    registry = get_registry()
    model_name = "test-unregister-model"

    # Test unregistering non-existent model
    with pytest.raises(ModelNotFoundError):
        registry.unregister("non-existent-model")

    # Register a test model
    registry.register(
        name=model_name,
        model_type=ModelType.GENERIC,
        model_class=AutoModel,
        config_class=AutoConfig,
    )

    # Load to ensure it's cached
    registry.load_model(model_name, trust_remote_code=True)
    
    # Verify it's registered and cached
    assert registry.get_metadata(model_name) is not None
    assert model_name in registry._model_cache  # type: ignore

    # Unregister
    registry.unregister(model_name)

    # Verify it's no longer registered
    with pytest.raises(ModelNotFoundError):
        registry.get_metadata(model_name)

    # Verify it's not in the cache
    assert model_name not in registry._model_cache  # type: ignore
    
    # Test unregistering with force=True even if not in registry
    registry.unregister("non-existent-model-2", force=True)


def test_cache_ttl():
    """Test time-based cache expiration."""
    registry = get_registry()
    model_name = TEST_MODELS[0]
    
    # Set a very short TTL (1 second)
    registry.set_cache_ttl(1)
    
    # Load the model (should be cached)
    model1 = registry.load_model(model_name, trust_remote_code=True)
    
    # Immediately load again (should be a cache hit)
    model2 = registry.load_model(model_name, trust_remote_code=True)
    assert model1 is model2
    
    # Wait for TTL to expire
    time.sleep(1.1)
    
    # Load again (should be a cache miss due to TTL)
    model3 = registry.load_model(model_name, trust_remote_code=True)
    assert model1 is not model3
    
    # Reset TTL to default
    registry.set_cache_ttl(3600)


def test_concurrent_registration():
    """Test concurrent model registration."""
    registry = get_registry()
    results = []
    lock = threading.Lock()
    
    def register_model(i):
        model_name = f"concurrent-reg-{i}"
        try:
            registry.register(
                name=model_name,
                model_type=ModelType.GENERIC,
                model_class=AutoModel,
                config_class=AutoConfig,
            )
            with lock:
                results.append(True)
        except Exception as e:
            with lock:
                results.append(False)
    
    # Create and start multiple threads
    num_threads = 10
    threads = [threading.Thread(target=register_model, args=(i,)) 
              for i in range(num_threads)]
    for t in threads:
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Verify all registrations were successful
    assert len(results) == num_threads
    assert all(results)
    
    # Verify all models are registered
    for i in range(num_threads):
        assert registry.get_metadata(f"concurrent-reg-{i}") is not None


def test_model_metadata():
    """Test model metadata functionality."""
    registry = get_registry()
    model_name = "metadata-test-model"
    
    # Register with custom metadata
    registry.register(
        name=model_name,
        model_type=ModelType.GENERIC,
        model_class=AutoModel,
        config_class=AutoConfig,
        description="Test model with metadata",
        version="1.0.0",
        tags=["test", "metadata"],
        custom_metadata={"author": "test", "license": "MIT"},
    )
    
    # Get metadata
    metadata = registry.get_metadata(model_name)
    
    # Verify metadata
    assert metadata.name == model_name
    assert metadata.model_type == ModelType.GENERIC
    assert metadata.description == "Test model with metadata"
    assert metadata.version == "1.0.0"
    assert set(metadata.tags) == {"test", "metadata"}
    assert hasattr(metadata, 'custom_metadata')
    assert metadata.custom_metadata["author"] == "test"
    assert metadata.custom_metadata["license"] == "MIT"
    assert hasattr(metadata, 'created_at')
    assert hasattr(metadata, 'updated_at')
    assert isinstance(metadata.created_at, datetime)
    assert isinstance(metadata.updated_at, datetime)
    assert metadata.load_count == 0
    
    # Update metadata
    registry.register(
        name=model_name,
        model_type=ModelType.GENERIC,
        model_class=AutoModel,
        config_class=AutoConfig,
        description="Updated description",
        version="2.0.0",
        tags=["test", "updated"],
    )
    
    # Verify metadata was updated
    updated_metadata = registry.get_metadata(model_name)
    assert updated_metadata.description == "Updated description"
    assert updated_metadata.version == "2.0.0"
    assert set(updated_metadata.tags) == {"test", "updated"}
    assert updated_metadata.created_at == metadata.created_at  # Should not change
    assert updated_metadata.updated_at > metadata.updated_at  # Should be updated
    assert updated_metadata.load_count == 0  # Should be preserved


if __name__ == "__main__":
    pytest.main([__file__])
