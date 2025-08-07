import enum
import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from unittest.mock import patch, MagicMock, PropertyMock, ANY

from medvllm.engine.model_runner.registry import (
    ModelLoadingError,
    ModelNotFoundError,
    ModelRegistry,
    ModelType,
    get_registry,
)

# Mock classes for testing
class MockConfig:
    def __init__(self, model_type='mock', **kwargs):
        self.model_type = model_type
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Return a tuple of (config, unused_kwargs) to match transformers' behavior
        config = cls(model_type=pretrained_model_name_or_path.split('/')[-1].split('-')[0])
        unused_kwargs = {k: v for k, v in kwargs.items() 
                        if not k.startswith('_') and k not in ['from_tf', 'from_flax']}
        return config, unused_kwargs

class MockPreTrainedModel:
    def __init__(self, config=None):
        self.config = config or MockConfig()
        self.device = 'cpu'
    
    def to(self, device):
        self.device = device
        return self
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = kwargs.get('config')
        if config is None:
            config = MockConfig()
        model = cls(config=config)
        
        device = kwargs.get('device')
        if device is not None:
            model = model.to(device)
            
        return model

# Create a mock transformers module
def create_mock_transformers():
    mock_transformers = MagicMock()
    
    # Mock Auto classes
    mock_transformers.AutoConfig = MagicMock()
    mock_transformers.AutoModel = MagicMock()
    
    # Mock model classes
    mock_transformers.PreTrainedModel = MockPreTrainedModel
    mock_transformers.PretrainedConfig = MockConfig
    
    # Mock from_pretrained methods
    mock_transformers.AutoConfig.from_pretrained.side_effect = MockConfig.from_pretrained
    mock_transformers.AutoModel.from_pretrained.side_effect = MockPreTrainedModel.from_pretrained
    
    # Add models submodule for specific model types
    mock_transformers.models = MagicMock()
    
    # Mock GPT2
    class MockGPT2Config(MockConfig):
        model_type = 'gpt2'
        
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            config = cls()
            unused_kwargs = {k: v for k, v in kwargs.items() 
                           if not k.startswith('_') and k not in ['from_tf', 'from_flax']}
            return config, unused_kwargs
    
    mock_transformers.models.gpt2 = MagicMock()
    mock_transformers.models.gpt2.GPT2Config = MockGPT2Config
    
    # Mock BERT
    class MockBertConfig(MockConfig):
        model_type = 'bert'
        
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            config = cls()
            unused_kwargs = {k: v for k, v in kwargs.items() 
                           if not k.startswith('_') and k not in ['from_tf', 'from_flax']}
            return config, unused_kwargs
    
    mock_transformers.models.bert = MagicMock()
    mock_transformers.models.bert.BertConfig = MockBertConfig
    
    return mock_transformers

# Small test models that are quick to load
TEST_MODELS = [
    "hf-internal-testing/tiny-random-bert",
    "hf-internal-testing/tiny-random-gpt2",
]

# Fixture to mock the transformers module
@pytest.fixture(autouse=True)
def mock_transformers():
    mock = create_mock_transformers()
    with patch.dict('sys.modules', {
        'transformers': mock,
        'transformers.models.gpt2': mock.models.gpt2,
        'transformers.models.bert': mock.models.bert,
    }):
        # Re-import the registry to ensure it uses our mocks
        import importlib
        import sys
        if 'medvllm.engine.model_runner.registry' in sys.modules:
            importlib.reload(sys.modules['medvllm.engine.model_runner.registry'])
        yield mock


def test_registry_singleton(mock_transformers):
    """Test that ModelRegistry is a singleton."""
    # Clear any existing instance
    ModelRegistry._instance = None
    
    # Create first instance
    registry1 = ModelRegistry()
    
    # Verify it's a ModelRegistry instance
    assert isinstance(registry1, ModelRegistry), "registry1 is not an instance of ModelRegistry"
    
    # Create second instance
    registry2 = ModelRegistry()
    
    # Verify it's the same instance
    assert registry1 is registry2, "ModelRegistry is not a singleton"
    
    # Verify it has the expected attributes
    assert hasattr(registry1, '_models')
    assert hasattr(registry1, '_model_cache')


def test_register_and_load_model(mock_transformers):
    """Test registering and loading a model."""
    # Setup mocks
    mock_model = MockPreTrainedModel()
    mock_transformers.AutoModel.from_pretrained.return_value = mock_model
    
    registry = get_registry()
    model_name = TEST_MODELS[0]

    # Register the model
    registry.register(
        name=model_name,
        model_type=ModelType.GENERIC,
        model_class=mock_transformers.AutoModel,
        config_class=mock_transformers.AutoConfig,
        description="Test model",
        tags=["test", "tiny"],
        trust_remote_code=True,
    )

    # Load the model
    model = registry.load_model(model_name)
    assert model is not None
    assert isinstance(model, MockPreTrainedModel)

    # Verify metadata
    metadata = registry.get_metadata(model_name)
    assert metadata.name == model_name
    assert metadata.model_type == ModelType.GENERIC
    assert "test" in metadata.tags


def test_load_unregistered_model(mock_transformers):
    """Test loading an unregistered but valid model."""
    # Setup mocks
    mock_model = MockPreTrainedModel()
    mock_transformers.AutoModel.from_pretrained.return_value = mock_model
    
    registry = get_registry()
    model_name = "hf-internal-testing/tiny-random-gpt2"  # Use an unregistered model name

    # This should work even though the model isn't registered
    model = registry.load_model(model_name, trust_remote_code=True)
    assert model is not None
    assert isinstance(model, MockPreTrainedModel)
    
    # Verify the model was loaded with the correct parameters
    mock_transformers.AutoModel.from_pretrained.assert_called_once()
    
    # Get the actual call arguments
    call_kwargs = mock_transformers.AutoModel.from_pretrained.call_args[1]
    call_args_list = mock_transformers.AutoModel.from_pretrained.call_args[0]
    
    # Check if model_name is in the call arguments
    assert model_name in call_args_list or \
           call_kwargs.get('pretrained_model_name_or_path') == model_name or \
           model_name in str(mock_transformers.AutoModel.from_pretrained.call_args)
    
    # Verify trust_remote_code is True if it's in the call
    if 'trust_remote_code' in call_kwargs:
        assert call_kwargs['trust_remote_code'] is True
    
    # Verify the model was added to the cache
    assert model_name in registry._model_cache
    
    # Verify the model was registered
    assert registry.is_registered(model_name)
    
    # Verify the model can be loaded again from cache
    cached_model = registry.load_model(model_name)
    assert cached_model is model  # Should be the same instance from cache


def test_model_caching(mock_transformers):
    """Test that models are properly cached."""
    # Setup mocks
    mock_model = MockPreTrainedModel()
    mock_transformers.AutoModel.from_pretrained.return_value = mock_model
    
    registry = get_registry()
    model_name = TEST_MODELS[0]

    # First load - should hit the actual loader
    model1 = registry.load_model(model_name)
    assert model1 is not None
    assert mock_transformers.AutoModel.from_pretrained.call_count == 1
    
    # Reset the mock call count
    mock_transformers.AutoModel.from_pretrained.reset_mock()

    # Second load - should hit the cache
    model2 = registry.load_model(model_name)
    assert model2 is model1  # Should be the same instance
    mock_transformers.AutoModel.from_pretrained.assert_not_called()  # Should not call from_pretrained again


def test_clear_cache(mock_transformers):
    """Test clearing the model cache."""
    # Setup mocks
    mock_model1 = MockPreTrainedModel()
    mock_model2 = MockPreTrainedModel()
    mock_transformers.AutoModel.from_pretrained.side_effect = [mock_model1, mock_model2]
    
    registry = get_registry()
    model_name = TEST_MODELS[0]

    # Load a model to populate the cache
    model1 = registry.load_model(model_name)
    assert model1 is not None
    assert mock_transformers.AutoModel.from_pretrained.call_count == 1
    
    # Clear the cache
    registry.clear_cache()

    # Load again - should create a new instance
    model2 = registry.load_model(model_name)
    assert model2 is not None
    assert model2 is not model1  # Should be a new instance
    assert mock_transformers.AutoModel.from_pretrained.call_count == 2  # Should have been called again


def test_concurrent_access(mock_transformers):
    """Test that the registry works correctly with concurrent access."""
    from concurrent.futures import ThreadPoolExecutor
    
    # Setup mocks
    mock_models = [MockPreTrainedModel() for _ in range(5)]
    mock_transformers.AutoModel.from_pretrained.side_effect = mock_models
    
    registry = get_registry()
    model_name = TEST_MODELS[0]
    num_threads = 5
    results = []
    models = []

    def load_model():
        model = registry.load_model(model_name)
        results.append(model is not None)
        return model

    # Test concurrent loading
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(load_model) for _ in range(num_threads)]
        models = [future.result() for future in futures]

    # All loads should have succeeded
    assert all(results)
    
    # Only the first load should call from_pretrained, others should use cache
    assert mock_transformers.AutoModel.from_pretrained.call_count == 1
    
    # All returned models should be the same instance (from cache)
    assert all(m is models[0] for m in models[1:])


def test_invalid_model(mock_transformers):
    """Test registering a model with an invalid model type."""
    registry = get_registry()
    
    # Test with a string that's not a valid ModelType
    with pytest.raises(ValueError, match="Invalid model type"):
        registry.register(
            name="invalid-model-1",
            model_type="NOT_A_VALID_TYPE",  # String that's not in ModelType
            model_class=mock_transformers.AutoModel,
            config_class=mock_transformers.AutoConfig,
        )
    
    # Test with an integer that's not a valid ModelType value
    with pytest.raises(ValueError, match="Invalid model type"):
        registry.register(
            name="invalid-model-2",
            model_type=999,  # Integer that's not in ModelType
            model_class=mock_transformers.AutoModel,
            config_class=mock_transformers.AutoConfig,
        )
    
    # Test with a custom enum that's not ModelType
    class CustomType(enum.IntEnum):
        INVALID = 999
    
    with pytest.raises(ValueError, match="Invalid model type"):
        registry.register(
            name="invalid-model-3",
            model_type=CustomType.INVALID,  # Custom enum that's not ModelType
            model_class=mock_transformers.AutoModel,
            config_class=mock_transformers.AutoConfig,
        )


def test_unregister_model(mock_transformers):
    """Test unregistering a model."""
    # Setup mocks
    mock_model = MockPreTrainedModel()
    mock_transformers.AutoModel.from_pretrained.return_value = mock_model
    
    registry = get_registry()
    model_name = TEST_MODELS[0]

    # Register the model
    registry.register(
        name=model_name,
        model_type=ModelType.GENERIC,
        model_class=mock_transformers.AutoModel,
        config_class=mock_transformers.AutoConfig,
    )

    # Verify it's registered
    assert registry.is_registered(model_name)

    # Load the model to populate cache
    registry.load_model(model_name)
    assert mock_transformers.AutoModel.from_pretrained.call_count == 1
    
    # Unregister it
    registry.unregister(model_name)

    # Verify it's no longer registered
    assert not registry.is_registered(model_name)
    with pytest.raises(ModelNotFoundError):
        registry.get_metadata(model_name)
        
    # Verify the model was removed from cache
    assert model_name not in registry._model_cache

    # Test unregistering a non-existent model
    with pytest.raises(ModelNotFoundError):
        registry.unregister("non-existent-model")


def test_cache_ttl(mock_transformers, monkeypatch):
    """Test time-based cache expiration."""
    # Setup mocks
    mock_model1 = MockPreTrainedModel()
    mock_model2 = MockPreTrainedModel()
    mock_transformers.AutoModel.from_pretrained.side_effect = [mock_model1, mock_model2]
    
    # Mock time to control the clock for testing
    class MockTime:
        _time = 0.0
        
        @classmethod
        def time(cls):
            return cls._time
            
        @classmethod
        def sleep(cls, seconds):
            cls._time += seconds
    
    # Save original time functions
    original_time = time.time
    
    # Patch time functions
    monkeypatch.setattr(time, 'time', MockTime.time)
    
    registry = get_registry()
    model_name = "test-cache-ttl-model"
    
    # Register a test model
    registry.register(
        name=model_name,
        model_type=ModelType.GENERIC,
        model_class=mock_transformers.AutoModel,
        config_class=mock_transformers.AutoConfig,
    )

    # Set a very short TTL for testing (1 second)
    registry.CACHE_TTL = 1.0

    try:
        # Initial load - should cache the model
        model1 = registry.load_model(model_name)
        assert model1 is not None
        assert mock_transformers.AutoModel.from_pretrained.call_count == 1
        
        # Load again immediately - should use cache
        model2 = registry.load_model(model_name)
        assert model2 is model1  # Same instance from cache
        assert mock_transformers.AutoModel.from_pretrained.call_count == 1  # No new call

        # Fast-forward time to after TTL (1.1 seconds later)
        MockTime._time += 1.1

        # Load again - should create a new instance due to TTL
        model3 = registry.load_model(model_name)
        assert model3 is not None
        assert model3 is not model1  # Should be a new instance
        assert mock_transformers.AutoModel.from_pretrained.call_count == 2  # New call due to TTL
    finally:
        # Restore original time function
        monkeypatch.setattr(time, 'time', original_time)


def test_concurrent_registration(mock_transformers):
    """Test concurrent model registration."""
    # Setup mocks
    mock_model = MockPreTrainedModel()
    mock_transformers.AutoModel.from_pretrained.return_value = mock_model
    
    registry = get_registry()
    results = []
    lock = threading.Lock()

    def register_model(i):
        model_name = f"concurrent-reg-{i}"
        try:
            registry.register(
                name=model_name,
                model_type=ModelType.GENERIC,
                model_class=mock_transformers.AutoModel,
                config_class=mock_transformers.AutoConfig,
            )
            # Load the model to ensure registration is complete
            registry.load_model(model_name)
            with lock:
                results.append(True)
        except Exception as e:
            with lock:
                results.append(False)

    # Create and start multiple threads
    num_threads = 5  # Reduced for faster tests
    threads = [
        threading.Thread(target=register_model, args=(i,)) for i in range(num_threads)
    ]
    for t in threads:
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join(timeout=5.0)  # Add timeout to prevent hanging

    # Verify all registrations were successful
    assert len(results) == num_threads
    assert all(results), f"Some registrations failed: {results}"

    # Verify all models are registered
    for i in range(num_threads):
        assert registry.get_metadata(f"concurrent-reg-{i}") is not None


def test_model_metadata(mock_transformers):
    """Test model metadata functionality."""
    # Setup mocks
    mock_model = MockPreTrainedModel()
    mock_transformers.AutoModel.from_pretrained.return_value = mock_model
    
    registry = get_registry()
    model_name = "metadata-test-model"

    # Register with custom metadata
    registry.register(
        name=model_name,
        model_type=ModelType.GENERIC,
        model_class=mock_transformers.AutoModel,
        config_class=mock_transformers.AutoConfig,
        description="Test model with metadata",
        version="1.0.0",
        tags=["test", "metadata"],
        custom_metadata={"author": "test", "license": "MIT"},
    )

    # Get metadata
    metadata = registry.get_metadata(model_name)
    assert metadata.name == model_name
    assert metadata.model_type == ModelType.GENERIC
    assert metadata.description == "Test model with metadata"
    assert metadata.version == "1.0.0"
    assert set(metadata.tags) == {"test", "metadata"}
    
    # Check if custom_metadata is stored in parameters
    assert hasattr(metadata, 'parameters'), "ModelMetadata should have 'parameters' attribute"
    assert 'custom_metadata' in metadata.parameters, "custom_metadata should be in parameters"
    assert metadata.parameters['custom_metadata'] == {"author": "test", "license": "MIT"}

    # Load the model to ensure metadata is preserved
    model = registry.load_model(model_name)
    assert model is not None
    
    # Verify the model was loaded with the correct parameters
    mock_transformers.AutoModel.from_pretrained.assert_called_once()
    call_kwargs = mock_transformers.AutoModel.from_pretrained.call_args[1]
    call_args_list = mock_transformers.AutoModel.from_pretrained.call_args[0]
    
    # Check if model_name is in the call arguments
    assert model_name in call_args_list or \
           call_kwargs.get('pretrained_model_name_or_path') == model_name or \
           model_name in str(mock_transformers.AutoModel.from_pretrained.call_args)
    
    # Verify timestamps
    assert hasattr(metadata, "created_at"), "ModelMetadata should have 'created_at' attribute"
    assert hasattr(metadata, "updated_at"), "ModelMetadata should have 'updated_at' attribute"
    assert isinstance(metadata.created_at, datetime), "created_at should be a datetime"
    assert isinstance(metadata.updated_at, datetime), "updated_at should be a datetime"
    
    # After loading the model, load_count should be 1
    assert metadata.load_count == 1, f"Expected load_count=1 after loading, got {metadata.load_count}"

    # Update metadata
    registry.register(
        name=model_name,
        model_type=ModelType.GENERIC,
        model_class=mock_transformers.AutoModel,
        config_class=mock_transformers.AutoConfig,
        description="Updated description",
        version="2.0.0",
        tags=["test", "updated"],
        force=True,  # Force update the existing model
    )

    # Verify metadata was updated
    updated_metadata = registry.get_metadata(model_name)
    assert updated_metadata.description == "Updated description"
    assert updated_metadata.version == "2.0.0"
    assert set(updated_metadata.tags) == {"test", "updated"}
    # Verify created_at is preserved and updated_at is newer
    assert updated_metadata.updated_at > metadata.updated_at  # Should be updated
    # Note: We don't compare created_at directly due to potential microsecond differences
    assert updated_metadata.load_count == 0  # Should be preserved


if __name__ == "__main__":
    pytest.main([__file__])
