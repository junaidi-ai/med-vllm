"""Tests for MedicalModelConfig compatibility with base Config."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Mock the imports that would normally be provided by the application
sys.modules['medvllm.medical.config.base'] = MagicMock()
sys.modules['medvllm.medical.config.base'].BaseMedicalConfig = type('BaseMedicalConfig', (), {})

# Mock transformers and other dependencies
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Now import the class we want to test
from medvllm.medical.config.medical_config import MedicalModelConfig


class TestMedicalConfigCompatibility:
    """Test compatibility of MedicalModelConfig with base Config."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = tmp_path
        self.mock_config = MagicMock()
        self.mock_config.model_type = "bert"
        self.mock_config.vocab_size = 30522
        self.mock_config.hidden_size = 768
        self.mock_config.num_hidden_layers = 12
        self.mock_config.num_attention_heads = 12
        self.mock_config.hidden_act = "gelu"
        
        # Patch transformers import
        self.patcher = patch("transformers.AutoConfig.from_pretrained", 
                           return_value=self.mock_config)
        self.mock_from_pretrained = self.patcher.start()
        
        yield
        
        # Cleanup
        self.patcher.stop()
    
    def test_base_config_compatibility(self):
        """Test that MedicalModelConfig works with base Config parameters."""
        base_params = {
            "model": "bert-base-uncased",
            "tokenizer": "bert-base-uncased",
            "trust_remote_code": False,
            "revision": "main",
            "max_seq_len": 2048,
            "dtype": "float16",
            "quantization": None,
            "enforce_eager": False,
            "max_context_len_to_capture": 8192,
            "max_logprobs": 5,
            "disable_log_stats": False,
            "download_dir": None,
            "load_format": "auto",
            "seed": 0,
            "worker_use_ray": False,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 1,
            "block_size": 16,
            "swap_space": 4,
            "gpu_memory_utilization": 0.9,
            "max_num_batched_tokens": None,
            "max_num_seqs": 256,
            "disable_log_requests": False,
            "max_log_len": None,
            "num_kvcache_blocks": 100,
            "model_type": "bert",
        }

        # Should work with just base parameters
        config = MedicalModelConfig(**base_params)

        # Verify base parameters are set correctly
        for key, value in base_params.items():
            assert getattr(config, key) == value, f"Mismatch for {key}"

    def test_medical_parameters(self):
        """Test that medical-specific parameters work correctly."""
        medical_params = {
            "model": "test-model",
            "max_medical_seq_length": 512,
            "medical_specialties": ["cardiology", "neurology"],
            "anatomical_regions": ["head", "chest"],
            "enable_uncertainty_estimation": True,
        }

        # Create config with medical parameters
        config = MedicalModelConfig(**medical_params)

        # Verify all parameters are set correctly
        for key, value in medical_params.items():
            assert getattr(config, key) == value, f"Mismatch for {key}"

    def test_serialization_roundtrip(self):
        """Test serialization/deserialization maintains all parameters."""
        # Create a config with all parameters
        all_params = {
            "model": "test-model",
            "max_seq_len": 2048,
            "dtype": "float16",
            "max_medical_seq_length": 512,
            "medical_specialties": ["cardiology", "neurology"],
            "anatomical_regions": ["head", "chest"],
            "enable_uncertainty_estimation": True,
        }

        # Create config and serialize to dict
        config = MedicalModelConfig(**all_params)
        config_dict = config.to_dict()

        # Create new config from serialized dict
        new_config = MedicalModelConfig.from_dict(config_dict)

        # Verify all parameters match
        for key, value in all_params.items():
            assert getattr(new_config, key) == value, f"Mismatch for {key} after roundtrip"

    def test_json_serialization(self, tmp_path):
        """Test JSON serialization/deserialization."""
        # Create a config with all parameters
        all_params = {
            "model": str(tmp_path / "test-model"),
            "max_seq_len": 2048,
            "dtype": "float16",
            "max_medical_seq_length": 512,
            "medical_specialties": ["cardiology", "neurology"],
            "anatomical_regions": ["head", "chest"],
            "enable_uncertainty_estimation": True,
        }

        # Create config and save to JSON
        config = MedicalModelConfig(**all_params)
        json_path = tmp_path / "config.json"
        config.to_json(json_path)

        # Load from JSON
        loaded_config = MedicalModelConfig.from_json(json_path)

        # Verify all parameters match
        for key, value in all_params.items():
            assert getattr(loaded_config, key) == value, f"Mismatch for {key} after JSON roundtrip"

    def test_backward_compatibility(self):
        """Test compatibility with base Config usage patterns."""
        # Test with base parameters
        base_params = {
            "model": "bert-base-uncased",
            "max_seq_len": 2048,
            "dtype": "float16",
        }

        # Should work with just base parameters
        config = MedicalModelConfig(**base_params)
        for key, value in base_params.items():
            assert getattr(config, key) == value, f"Mismatch for {key}"

        # Test with medical parameters
        medical_params = {
            "max_medical_seq_length": 512,
            "medical_specialties": ["cardiology", "neurology"],
            "anatomical_regions": ["head", "chest"],
            "enable_uncertainty_estimation": True,
        }

        # Create config with medical parameters
        full_config = {**base_params, "model": "test-model", **medical_params}
        config = MedicalModelConfig(**full_config)

        # Verify all parameters are set correctly
        for key, value in full_config.items():
            assert getattr(config, key) == value, f"Mismatch for {key}"

        # Test with from_dict
        config_from_dict = MedicalModelConfig.from_dict(full_config)
        for key, value in full_config.items():
            assert (
                getattr(config_from_dict, key) == value
            ), f"Mismatch for {key} in from_dict"
