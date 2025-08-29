"""Unit tests for ModelRegistry using the real implementation.

This test avoids clobbering sys.modules to prevent cross-test interference.
"""

from unittest.mock import MagicMock

import pytest

from medvllm.engine.model_runner.registry import ModelRegistry, ModelType
from medvllm.engine.model_runner.exceptions import ModelNotFoundError


class TestModelRegistryCore:
    """Test core functionality of ModelRegistry."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset the singleton instance and internal state
        ModelRegistry._instance = None  # type: ignore[attr-defined]
        self.registry = ModelRegistry()
        self.registry._models = {}
        self.registry._model_cache = {}

        # Create mock model and a real config class (avoid MagicMock __name__ issues)
        self.mock_model = MagicMock()
        self.mock_model.to = MagicMock(return_value=self.mock_model)

        class DummyConfig:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        self.mock_config = DummyConfig

        # Use a concrete dummy model class with a real __name__ to satisfy metadata
        class DummyModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):  # will be patched below
                raise NotImplementedError

        # Patch classmethod to return our mock model and allow call assertion
        DummyModel.from_pretrained = MagicMock(return_value=self.mock_model)
        self.mock_model_cls = DummyModel

        # Register a test model
        self.test_model_name = "test/model"
        self.registry.register(
            name=self.test_model_name,
            # Register as GENERIC to avoid medical adapter load path during unit test
            model_type=ModelType.GENERIC,
            model_class=self.mock_model_cls,
            config_class=self.mock_config,
            description="Test model",
            tags=["test", "unit"],
        )

    def test_register_and_retrieve_model(self):
        """Test registering and retrieving a model from the registry."""
        # Verify the model was registered
        metadata = self.registry.get_metadata(self.test_model_name)
        assert metadata is not None
        assert metadata.name == self.test_model_name
        assert metadata.model_type == ModelType.GENERIC
        assert metadata.model_class == self.mock_model_cls
        assert metadata.description == "Test model"
        assert "test" in metadata.tags
        assert "unit" in metadata.tags

    def test_list_models(self):
        """Test listing registered models."""
        models = self.registry.list_models()
        assert len(models) == 1
        assert models[0]["name"] == self.test_model_name
        assert models[0]["model_type"] == ModelType.GENERIC.name

    def test_load_model(self):
        """Test loading a model through the registry."""
        device = "cpu"
        model = self.registry.load_model(self.test_model_name, device=device)

        # Verify the model was loaded correctly
        assert model == self.mock_model
        # Verify our loader was invoked with the model name
        assert self.mock_model_cls.from_pretrained.called
        args, kwargs = self.mock_model_cls.from_pretrained.call_args
        assert args[0] == self.test_model_name

    def test_duplicate_registration(self):
        """Duplicate registrations should raise ValueError and not alter existing entry."""
        with pytest.raises(ValueError):
            self.registry.register(
                name=self.test_model_name,
                model_type=ModelType.CLINICAL,  # Different type
                model_class=MagicMock(),  # Different class
                config_class=MagicMock(),  # Different config
            )

        # Verify the original registration is still intact
        metadata = self.registry.get_metadata(self.test_model_name)
        assert metadata.model_type == ModelType.GENERIC
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
        with pytest.raises(ModelNotFoundError):
            self.registry.get_metadata(self.test_model_name)

        # Verify the model is not in the list
        assert len(self.registry.list_models()) == 0
