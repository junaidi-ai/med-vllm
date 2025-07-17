"""Pytest configuration for performance tests."""
import pytest
from unittest.mock import MagicMock
from tests.performance.test_utils import ModelType, ModelMetadata, TestModel, TestConfig


class ModelRegistry:
    """A simplified ModelRegistry for performance testing."""
    def __init__(self):
        self._models = {}
        self._model_cache = {}
        self._cache_size = 3
        self._lock = MagicMock()
        self._model_counter = 0

    def register(self, name, model_type, model_class, config_class, description="", tags=None, **parameters):
        """Register a model."""
        self._models[name] = ModelMetadata(
            name=name,
            model_type=model_type,
            model_class=model_class,
            config_class=config_class,
            description=description,
            tags=tags or [],
            parameters=parameters or {}
        )
        return self._models[name]

    def load_model(self, name, use_cache=True):
        """Load a model, with optional caching."""
        if name not in self._models:
            raise ValueError(f"Model {name} not found")
        
        if use_cache and name in self._model_cache:
            return self._model_cache[name]
        
        # Simulate model loading with a small delay to simulate real loading
        import time
        time.sleep(0.01)  # 10ms delay to simulate model loading
        
        metadata = self._models[name]
        model = TestModel(name=name)  # Create a test model instance
        
        if use_cache:
            # Simple LRU cache implementation
            if len(self._model_cache) >= self._cache_size:
                self._model_cache.pop(next(iter(self._model_cache)))
            self._model_cache[name] = model
        
        return model

    def unload_model(self, name):
        """Unload a model from the cache."""
        self._model_cache.pop(name, None)

    def clear(self):
        """Clear all models and cache."""
        self._models.clear()
        self._model_cache.clear()

    def list_models(self):
        """List all registered models."""
        return list(self._models.keys())


@pytest.fixture(scope="module")
def model_registry():
    """Fixture providing a ModelRegistry instance for testing."""
    # Create a test registry with a small cache size
    registry = ModelRegistry()
    
    # Register some test models
    for i in range(5):
        model_name = f"test-model-{i}"
        registry._models[model_name] = ModelMetadata(
            name=model_name,
            model_type=ModelType.GENERIC,
            model_class=TestModel,
            config_class=TestConfig,
            description=f"Test model {i}",
            tags=["test", f"model-{i}"],
            parameters={"test_param": i}
        )
    
    yield registry
    
    # Cleanup
    registry.clear()


@pytest.fixture
def temp_model_dir(tmp_path):
    """Fixture providing a temporary directory for model storage."""
    return str(tmp_path / "models")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "performance: mark test as a performance test"
    )
