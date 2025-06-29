"""
Performance tests for configuration handling.

This module contains performance benchmarks for configuration loading,
validation, and serialization.
"""

import json
from pathlib import Path
import tempfile
from typing import Dict, Any

import pytest

# Import the actual implementation
from medvllm.medical.config import MedicalModelConfig

# Sample configuration data for performance testing
SAMPLE_CONFIG = {
    "model_type": "medical_llm",
    "model_name_or_path": "medical-bert-base",
    "vocab_size": 30522,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "use_cache": True,
    "classifier_dropout": 0.1,
    "medical_specialties": ["cardiology", "radiology"],
    "anatomical_regions": ["head", "chest"],
    "max_sequence_length": 512,
}

# Large configuration for stress testing
LARGE_CONFIG = {
    **SAMPLE_CONFIG,
    "model_name_or_path": "large-medical-model",
    "hidden_size": 4096,
    "num_hidden_layers": 48,
    "num_attention_heads": 64,
    "intermediate_size": 16384,
    "medical_specialties": [f"specialty_{i}" for i in range(50)],
    "anatomical_regions": [f"region_{i}" for i in range(100)],
    "additional_config": {
        f"key_{i}": f"value_{i}" for i in range(1000)
    }
}


class TestConfigPerformance:
    """Performance tests for configuration handling."""
    
    @pytest.mark.performance
    def test_config_creation(self, benchmark) -> None:
        """Benchmark configuration object creation."""
        benchmark(MedicalModelConfig, **SAMPLE_CONFIG)
    
    @pytest.mark.performance
    def test_config_validation(self, benchmark) -> None:
        """Benchmark configuration validation."""
        config = MedicalModelConfig(**SAMPLE_CONFIG)
        benchmark(config.validate)
    
    @pytest.mark.performance
    def test_config_serialization(self, benchmark) -> None:
        """Benchmark configuration serialization to dict."""
        config = MedicalModelConfig(**SAMPLE_CONFIG)
        benchmark(config.to_dict)
    
    @pytest.mark.performance
    def test_config_deserialization(self, benchmark) -> None:
        """Benchmark configuration deserialization from dict."""
        benchmark(MedicalModelConfig.from_dict, SAMPLE_CONFIG)
    
    @pytest.mark.performance
    def test_large_config_creation(self, benchmark) -> None:
        """Benchmark creation of large configuration objects."""
        benchmark(MedicalModelConfig, **LARGE_CONFIG)
    
    @pytest.mark.performance
    def test_config_save_and_load(self, benchmark, tmp_path: Path) -> None:
        """Benchmark saving and loading configuration to/from disk."""
        config = MedicalModelConfig(**SAMPLE_CONFIG)
        
        def save_and_load() -> None:
            config_path = tmp_path / "config.json"
            config.save_pretrained(tmp_path)
            MedicalModelConfig.from_pretrained(tmp_path)
        
        benchmark(save_and_load)
    
    @pytest.mark.performance
    def test_config_validation_large(self, benchmark) -> None:
        """Benchmark validation of large configurations."""
        config = MedicalModelConfig(**LARGE_CONFIG)
        benchmark(config.validate)
    
    @pytest.mark.performance
    def test_config_serialization_large(self, benchmark) -> None:
        """Benchmark serialization of large configurations."""
        config = MedicalModelConfig(**LARGE_CONFIG)
        benchmark(config.to_dict)
    
    @pytest.mark.performance
    def test_config_deserialization_large(self, benchmark) -> None:
        """Benchmark deserialization of large configurations."""
        benchmark(MedicalModelConfig.from_dict, LARGE_CONFIG)
    
    @pytest.mark.performance
    def test_config_copy(self, benchmark) -> None:
        """Benchmark copying configuration objects."""
        config = MedicalModelConfig(**SAMPLE_CONFIG)
        benchmark(config.copy)
    
    @pytest.mark.performance
    def test_config_equality(self, benchmark) -> None:
        """Benchmark configuration equality comparison."""
        config1 = MedicalModelConfig(**SAMPLE_CONFIG)
        config2 = MedicalModelConfig(**SAMPLE_CONFIG)
        benchmark(lambda: config1 == config2)
    
    @pytest.mark.performance
    def test_config_hashing(self, benchmark) -> None:
        """Benchmark hashing of configuration objects."""
        config = MedicalModelConfig(**SAMPLE_CONFIG)
        benchmark(hash, config)
    
    @pytest.mark.performance
    def test_config_to_json(self, benchmark) -> None:
        """Benchmark conversion to JSON string."""
        config = MedicalModelConfig(**SAMPLE_CONFIG)
        benchmark(config.to_json_string)
    
    @pytest.mark.performance
    def test_config_to_json_large(self, benchmark) -> None:
        """Benchmark conversion of large config to JSON string."""
        config = MedicalModelConfig(**LARGE_CONFIG)
        benchmark(config.to_json_string)


@pytest.fixture(scope="module")
def large_config_file() -> Path:
    """Create a temporary file with a large configuration for testing."""
    config = MedicalModelConfig(**LARGE_CONFIG)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(config.to_json_string())
    return Path(f.name)


def test_config_loading_from_large_file(benchmark, large_config_file: Path) -> None:
    """Benchmark loading configuration from a large file."""
    def load_config() -> None:
        MedicalModelConfig.from_pretrained(large_config_file.parent)
    
    benchmark(load_config)
    
    # Cleanup
    large_config_file.unlink()
