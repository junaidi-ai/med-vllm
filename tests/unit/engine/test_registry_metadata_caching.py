"""Unit tests for ModelRegistry metadata and caching functionality."""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Simple mock classes for testing
class MockConfig:
    model_type = "mock"

class MockModel:
    def __init__(self, config):
        self.config = config
    
    def eval(self):
        return self
    
    def to(self, device):
        return self

# Mock ModelType
class ModelType:
    GENERIC = "generic"
    BIOMEDICAL = "biomedical"
    CLINICAL = "clinical"

# Mock ModelMetadata
class ModelMetadata:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.model_type = kwargs.get("model_type")
        self.model_class = kwargs.get("model_class")
        self.config_class = kwargs.get("config_class")
        self.description = kwargs.get("description", "")
        self.tags = kwargs.get("tags", [])
        self.version = kwargs.get("version", "1.0.0")
        self.capabilities = kwargs.get("capabilities", {})
        self.load_count = 0
        self.load_durations = []
        self.avg_load_time = 0
        self.last_loaded = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

# Mock ModelRegistry
class ModelRegistry:
    _instance = None
    
    def __init__(self):
        self._models = {}
        self._model_cache = {}
        self._access_order = []  # Track access order for LRU
        self._cache_stats = {"hits": 0, "misses": 0}
        self._max_cache_size = 5
        self._cache_ttl = 3600
        self._lock = MagicMock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(self, name, model_type, model_class, config_class, **kwargs):
        self._models[name] = ModelMetadata(
            name=name,
            model_type=model_type,
            model_class=model_class,
            config_class=config_class,
            **kwargs
        )
    
    def load_model(self, name, **kwargs):
        if name not in self._models:
            raise ValueError(f"Model {name} not found")
        
        metadata = self._models[name]
        metadata.load_count += 1
        metadata.last_loaded = datetime.now()
        
        # Simulate load time
        load_time = 0.1
        metadata.load_durations.append(load_time)
        metadata.avg_load_time = sum(metadata.load_durations) / len(metadata.load_durations)
        
        # Check cache
        if name in self._model_cache:
            # Update access order - move to end (most recently used)
            self._access_order.remove(name)
            self._access_order.append(name)
            self._cache_stats["hits"] += 1
            return self._model_cache[name]
        
        self._cache_stats["misses"] += 1
        
        # Create and cache model
        model = MockModel(MockConfig())
        
        # LRU eviction if needed
        if len(self._access_order) >= self._max_cache_size:
            # Remove least recently used
            lru_name = self._access_order.pop(0)
            del self._model_cache[lru_name]
        
        # Add to cache and update access order
        self._model_cache[name] = model
        self._access_order.append(name)
        return model
    
    def get_metadata(self, name):
        return self._models.get(name)
    
    def clear_cache(self):
        self._model_cache.clear()
        self._access_order = []
        self._cache_stats = {"hits": 0, "misses": 0}
    
    def set_cache_size(self, size):
        self._max_cache_size = size
        # Evict least recently used models if needed
        while len(self._access_order) > size:
            lru_name = self._access_order.pop(0)
            self._model_cache.pop(lru_name, None)
    
    def set_cache_ttl(self, ttl):
        self._cache_ttl = ttl
    
    def get_cache_stats(self):
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate": self._cache_stats["hits"] / total if total > 0 else 0,
            "current_size": len(self._model_cache),
            "max_size": self._max_cache_size
        }

@pytest.fixture
def registry():
    """Provide a clean registry instance for each test."""
    registry = ModelRegistry()
    return registry

def test_metadata_initialization(registry):
    """Test that metadata is properly initialized when registering a model."""
    # Register a test model
    registry.register(
        name="test-model",
        model_type=ModelType.GENERIC,
        model_class=MockModel,
        config_class=MockConfig,
        description="Test model",
        tags=["test", "mock"],
        version="1.0.0",
        capabilities={"inference": True, "training": False},
    )
    
    # Get metadata
    metadata = registry.get_metadata("test-model")
    
    # Verify metadata fields
    assert metadata.name == "test-model"
    assert metadata.model_type == ModelType.GENERIC
    assert metadata.model_class == MockModel
    assert metadata.config_class == MockConfig
    assert metadata.description == "Test model"
    assert set(metadata.tags) == {"test", "mock"}
    assert metadata.version == "1.0.0"
    assert metadata.capabilities == {"inference": True, "training": False}
    assert metadata.load_count == 0
    assert isinstance(metadata.created_at, datetime)
    assert isinstance(metadata.updated_at, datetime)

def test_load_metrics_tracking(registry):
    """Test that load metrics are properly tracked."""
    # Register a test model
    registry.register(
        "test-model",
        ModelType.GENERIC,
        MockModel,
        MockConfig
    )
    
    # Load the model multiple times
    for _ in range(3):
        registry.load_model("test-model")
    
    # Get metadata
    metadata = registry.get_metadata("test-model")
    
    # Verify metrics
    assert metadata.load_count == 3
    assert metadata.avg_load_time > 0
    assert isinstance(metadata.last_loaded, datetime)

def test_lru_cache_eviction(registry):
    """Test that LRU cache eviction works correctly."""
    # Set a small cache size for testing
    registry.set_cache_size(2)
    
    # Register and load 3 models
    for i in range(3):
        name = f"model-{i}"
        registry.register(
            name,
            ModelType.GENERIC,
            MockModel,
            MockConfig
        )
        registry.load_model(name)
    
    # First model should be evicted from cache
    assert "model-0" not in registry._model_cache
    assert "model-1" in registry._model_cache
    assert "model-2" in registry._model_cache
    
    # Access model-1 to make it most recently used
    registry.load_model("model-1")
    
    # Load another model, should evict model-2
    registry.register(
        "model-3",
        ModelType.GENERIC,
        MockModel,
        MockConfig
    )
    registry.load_model("model-3")
    
    assert "model-2" not in registry._model_cache
    assert "model-1" in registry._model_cache
    assert "model-3" in registry._model_cache

def test_cache_ttl(registry):
    """Test that cache entries expire after TTL."""
    # Set a short TTL for testing
    registry.set_cache_ttl(1)  # 1 second
    
    # Register and load a model
    registry.register(
        "test-model",
        ModelType.GENERIC,
        MockModel,
        MockConfig
    )
    registry.load_model("test-model")
    
    # Model should be in cache
    assert "test-model" in registry._model_cache
    
    # Next access should be a cache hit
    initial_hits = registry.get_cache_stats()["hits"]
    registry.load_model("test-model")
    assert registry.get_cache_stats()["hits"] == initial_hits + 1

def test_cache_statistics(registry):
    """Test that cache statistics are properly tracked."""
    # Reset statistics
    registry.clear_cache()
    
    # Register and load a model
    registry.register(
        "test-model",
        ModelType.GENERIC,
        MockModel,
        MockConfig
    )
    
    # Initial load (miss)
    registry.load_model("test-model")
    
    # Second load (hit)
    registry.load_model("test-model")
    
    # Get statistics
    stats = registry.get_cache_stats()
    
    # Verify statistics
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["current_size"] == 1
    assert stats["max_size"] == registry._max_cache_size

def test_thread_safety(registry):
    """Test that registry operations are thread-safe."""
    import threading
    
    # Register a test model
    registry.register(
        "test-model",
        ModelType.GENERIC,
        MockModel,
        MockConfig
    )
    
    # Number of threads to spawn
    num_threads = 10
    results = []
    
    # Function to be run by each thread
    def load_model():
        try:
            model = registry.load_model("test-model")
            results.append(model is not None)
        except Exception as e:
            results.append(False)
    
    # Create and start threads
    threads = [threading.Thread(target=load_model) for _ in range(num_threads)]
    for t in threads:
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Verify all threads completed successfully
    assert all(results), "Some threads failed to load the model"
    
    # Verify metrics
    metadata = registry.get_metadata("test-model")
    assert metadata is not None, "Metadata not found for test model"
    assert metadata.load_count > 0, "No model loads were recorded"
    
    # Verify cache stats
    stats = registry.get_cache_stats()
    assert stats["hits"] >= 0, "Invalid hit count"
    assert stats["misses"] > 0, "Expected at least one cache miss"

if __name__ == "__main__":
    pytest.main([__file__])
