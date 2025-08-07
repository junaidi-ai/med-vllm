"""Core tests for the ModelRegistry class."""

import os

# Import the registry module directly to avoid dependency issues
import sys
import threading
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from medvllm.engine.model_runner.registry import ModelMetadata, ModelRegistry, ModelType


class TestModel:
    """A simple mock model class for testing."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class TestConfig:
    """A simple mock config class for testing."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def test_registry_initialization():
    """Test that the registry initializes correctly."""
    registry = ModelRegistry()
    assert registry is not None
    assert isinstance(registry, ModelRegistry)


def test_register_model():
    """Test registering a model with the registry."""
    registry = ModelRegistry()
    registry._models = {}  # Reset models for test isolation

    # Register a test model
    registry.register(
        name="test-model",
        model_type=ModelType.BIOMEDICAL,
        model_class=TestModel,
        config_class=TestConfig,
        description="A test model",
        tags=["test", "mock"],
    )

    # Verify the model was registered
    assert "test-model" in registry._models
    metadata = registry._models["test-model"]
    assert metadata.name == "test-model"
    assert metadata.model_type == ModelType.BIOMEDICAL
    assert "test" in metadata.tags
    assert "mock" in metadata.tags


def test_duplicate_registration():
    """Test that registering a duplicate model raises an error."""
    registry = ModelRegistry()
    registry._models = {}  # Reset models for test isolation

    # Register a test model
    registry.register(
        name="test-dup",
        model_type=ModelType.BIOMEDICAL,
        model_class=TestModel,
        config_class=TestConfig,
    )

    # Try to register the same model again
    with pytest.raises(ValueError):
        registry.register(
            name="test-dup",
            model_type=ModelType.BIOMEDICAL,
            model_class=TestModel,
            config_class=TestConfig,
        )


def test_get_metadata():
    """Test retrieving model metadata."""
    registry = ModelRegistry()
    registry._models = {}  # Reset models for test isolation

    # Register a test model
    registry.register(
        name="test-meta",
        model_type=ModelType.BIOMEDICAL,
        model_class=TestModel,
        config_class=TestConfig,
        description="Metadata test",
    )

    # Get metadata
    metadata = registry.get_metadata("test-meta")
    assert metadata.name == "test-meta"
    assert metadata.description == "Metadata test"


def test_get_nonexistent_metadata():
    """Test that getting metadata for a non-existent model raises an error."""
    from medvllm.engine.model_runner.exceptions import ModelNotFoundError
    
    registry = ModelRegistry()
    registry._models = {}  # Reset models for test isolation

    with pytest.raises(ModelNotFoundError):
        registry.get_metadata("nonexistent-model")


def test_unregister_model():
    """Test unregistering a model."""
    registry = ModelRegistry()
    registry._models = {}  # Reset models for test isolation

    # Register a test model
    registry.register(
        name="test-unreg",
        model_type=ModelType.BIOMEDICAL,
        model_class=TestModel,
        config_class=TestConfig,
    )

    # Unregister the model
    registry.unregister("test-unreg")

    # Verify the model was unregistered
    assert "test-unreg" not in registry._models


def test_thread_safety():
    """Test that the registry is thread-safe."""
    registry = ModelRegistry()
    registry._models = {}  # Reset models for test isolation

    # Number of threads to create
    num_threads = 10
    results = []

    def register_model(thread_id):
        """Helper function to register a model from a thread."""
        try:
            model_name = f"thread-{thread_id}"
            registry.register(
                name=model_name,
                model_type=ModelType.BIOMEDICAL,
                model_class=TestModel,
                config_class=TestConfig,
            )
            results.append(True)
        except Exception:
            results.append(False)

    # Create and start threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=register_model, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Verify all registrations were successful
    assert all(results)

    # Verify all models were registered
    for i in range(num_threads):
        model_name = f"thread-{i}"
        assert model_name in registry._models


if __name__ == "__main__":
    pytest.main([__file__])
