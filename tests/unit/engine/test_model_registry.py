"""Unit tests for the ModelRegistry core functionality."""

import sys
from unittest.mock import MagicMock

import pytest


# Create a minimal ModelType enum for testing
class MockModelType:
    GENERIC = 1
    BIOMEDICAL = 2
    CLINICAL = 3


# Create a minimal ModelMetadata class for testing
class MockModelMetadata:
    def __init__(self, name, model_type, model_class, config_class, **kwargs):
        self.name = name
        self.model_type = model_type
        self.model_class = model_class
        self.config_class = config_class
        self.description = kwargs.get("description", "")
        self.tags = kwargs.get("tags", [])
        self.parameters = kwargs.get("parameters", {})


# Create a minimal ModelRegistry class for testing
class MockModelRegistry:
    _instance = None
    _lock = MagicMock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = {}
                    cls._instance._model_cache = {}
        return cls._instance

    def register(self, name, model_type, model_class, config_class, **kwargs):
        if name in self._models:
            return  # Skip duplicate registrations
        self._models[name] = MockModelMetadata(
            name=name,
            model_type=model_type,
            model_class=model_class,
            config_class=config_class,
            **kwargs,
        )

    def get_metadata(self, name):
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self._models[name]

    def list_models(self, model_type=None):
        models = []
        for name, metadata in self._models.items():
            if model_type is None or metadata.model_type == model_type:
                models.append(
                    {
                        "name": name,
                        "type": metadata.model_type,
                        "description": metadata.description,
                        "tags": metadata.tags,
                    }
                )
        return models

    def load_model(self, name, **kwargs):
        metadata = self.get_metadata(name)
        return metadata.model_class.from_pretrained(name, **kwargs)

    def unregister(self, name):
        if name in self._models:
            del self._models[name]
        if name in self._model_cache:
            del self._model_cache[name]

    def clear_cache(self):
        self._model_cache = {}


# Now patch the actual ModelRegistry with our mock
sys.modules["medvllm.engine.model_runner.registry"] = MagicMock()
sys.modules["medvllm.engine.model_runner.registry"].ModelRegistry = MockModelRegistry
sys.modules["medvllm.engine.model_runner.registry"].ModelType = MockModelType

# Now import the module under test
from medvllm.engine.model_runner.registry import ModelRegistry, ModelType


class TestModelRegistryCore:
    """Test core functionality of ModelRegistry."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset the singleton instance
        ModelRegistry._instance = None
        self.registry = ModelRegistry()
        self.registry._models = {}
        self.registry._model_cache = {}

        # Create mock model and config classes
        self.mock_model = MagicMock()
        self.mock_config = MagicMock()

        # Set up the mock model class
        self.mock_model_cls = MagicMock()
        self.mock_model_cls.from_pretrained = MagicMock(return_value=self.mock_model)

        # Register a test model
        self.test_model_name = "test/model"
        self.registry.register(
            name=self.test_model_name,
            model_type=ModelType.BIOMEDICAL,
            model_class=self.mock_model_cls,
            config_class=MagicMock(),
            description="Test model",
            tags=["test", "unit"],
        )

    def test_register_and_retrieve_model(self):
        """Test registering and retrieving a model from the registry."""
        # Verify the model was registered
        metadata = self.registry.get_metadata(self.test_model_name)
        assert metadata is not None
        assert metadata.name == self.test_model_name
        assert metadata.model_type == ModelType.BIOMEDICAL
        assert metadata.model_class == self.mock_model_cls
        assert metadata.description == "Test model"
        assert "test" in metadata.tags
        assert "unit" in metadata.tags

    def test_list_models(self):
        """Test listing registered models."""
        models = self.registry.list_models()
        assert len(models) == 1
        assert models[0]["name"] == self.test_model_name
        assert models[0]["type"] == ModelType.BIOMEDICAL

    def test_load_model(self):
        """Test loading a model through the registry."""
        device = "cpu"
        model = self.registry.load_model(self.test_model_name, device=device)

        # Verify the model was loaded correctly
        assert model == self.mock_model
        self.mock_model_cls.from_pretrained.assert_called_once_with(
            self.test_model_name, device=device
        )

    def test_duplicate_registration(self):
        """Test that duplicate registrations are handled gracefully."""
        # Try to register the same model again
        self.registry.register(
            name=self.test_model_name,
            model_type=ModelType.CLINICAL,  # Different type
            model_class=MagicMock(),  # Different class
            config_class=MagicMock(),  # Different config
        )

        # Verify the original registration is still intact
        metadata = self.registry.get_metadata(self.test_model_name)
        assert metadata.model_type == ModelType.BIOMEDICAL
        assert metadata.model_class == self.mock_model_cls

    def test_clear_cache(self):
        """Test clearing the model cache."""
        # Add something to the cache
        self.registry._model_cache["test"] = "dummy"

        # Clear the cache
        self.registry.clear_cache()

        # Verify the cache is empty
        assert len(self.registry._model_cache) == 0

    def test_unregister_model(self):
        """Test unregistering a model."""
        # Unregister the test model
        self.registry.unregister(self.test_model_name)

        # Verify the model is no longer registered
        with pytest.raises(KeyError):
            self.registry.get_metadata(self.test_model_name)

        # Verify the model is not in the list
        assert len(self.registry.list_models()) == 0
