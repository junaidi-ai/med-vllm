"""Integration tests for ModelRegistry with real dependencies."""

import pytest


@pytest.mark.integration
class TestModelRegistryIntegration:
    """Integration tests for ModelRegistry with real dependencies."""

    def test_register_and_load_model(
        self, model_registry, mock_model_config, mock_tokenizer
    ):
        """Test registering and loading a model with the registry."""
        # Register a test model
        model_registry.register(
            name="test-model-integration",
            model_type="GENERIC",
            model_class=mock_tokenizer.__class__,
            description="Integration test model",
            tags=["integration", "test"],
        )

        # Verify the model was registered
        models = model_registry.list_models()
        assert any(m.name == "test-model-integration" for m in models)

        # Test getting model metadata
        metadata = model_registry.get_metadata("test-model-integration")
        assert metadata is not None
        assert metadata.name == "test-model-integration"
        assert "integration" in metadata.tags

        # Test loading the model
        loaded_model = model_registry.load_model("test-model-integration")
        assert loaded_model is not None

        # Test model inference
        inputs = "Test input text"
        outputs = loaded_model(inputs)
        assert "input_ids" in outputs
        assert "attention_mask" in outputs

    def test_model_config_loading(self, model_registry):
        """Test loading model configurations."""
        # Test getting config for a known model type
        config = model_registry.get_model_config("qwen-test")
        assert config is not None
        assert hasattr(config, "model_type")
        assert hasattr(config, "vocab_size")

    def test_model_listing(self, model_registry):
        """Test listing registered models with filters."""
        # Clear any existing models
        for model in model_registry.list_models():
            model_registry.unregister(model.name)

        # Register test models
        model_registry.register(
            name="model-a",
            model_type="GENERIC",
            model_class=object,
            description="Test model A",
        )

        model_registry.register(
            name="model-b",
            model_type="BIOMEDICAL",
            model_class=object,
            description="Test model B",
        )

        # Test listing all models
        all_models = model_registry.list_models()
        assert len(all_models) == 2

        # Test filtering by model type
        bio_models = model_registry.list_models(model_type="BIOMEDICAL")
        assert len(bio_models) == 1
        assert bio_models[0].name == "model-b"
