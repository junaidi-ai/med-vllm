"""Tests for MedicalModelConfig class and related functionality."""

import os
import tempfile
import pytest
from pathlib import Path
from typing import Dict, List, Union
from unittest.mock import patch, MagicMock

# Import only what we need for testing
from medvllm.medical.config.medical_config import MedicalModelConfig
from medvllm.medical.config.serialization import ConfigSerializer
from medvllm.medical.config.schema import MedicalModelConfigSchema

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit

# Fixtures
@pytest.fixture
def sample_config() -> Dict:
    """Return a sample configuration dictionary for testing."""
    return {
        "model": "gpt2",
        "tensor_parallel_size": 1,
        "max_seq_len": 1024,
        "max_batch_size": 8,
        "dtype": "float16",
        "quantization": None,
        "enforce_eager": False,
        "max_context_len_to_capture": 8192,
        "max_logprobs": 5,
        "disable_log_stats": False,
        "revision": None,
        "code_revision": None,
        "tokenizer_revision": None,
        "trust_remote_code": False,
        "download_dir": None,
        "load_format": "auto",
        "seed": 0,
        "worker_use_ray": False,
        "pipeline_parallel_size": 1,
        "block_size": 16,
        "swap_space": 4,
        "gpu_memory_utilization": 0.90,
        "max_num_batched_tokens": None,
        "max_num_seqs": 256,
        "disable_log_requests": False,
        "max_log_len": None,
        "medical_specialties": ["cardiology", "neurology"],
        "anatomical_regions": ["head", "chest"],
        "config_version": "0.1.0"
    }


class TestMedicalModelConfigBasic:
    """Basic tests for MedicalModelConfig core functionality."""
    
    def test_medical_model_config_creation(self):
        """Test basic MedicalModelConfig creation."""
        config = MedicalModelConfig(
            model="test-model",
            medical_specialties=["cardiology"],
            anatomical_regions=["head"]
        )
        
        assert config.model == "test-model"
        assert config.medical_specialties == ["cardiology"]
        assert config.anatomical_regions == ["head"]

    def test_medical_specialties_validation(self):
        """Test validation of medical_specialties field."""
        # Test with string
        config = MedicalModelConfig(medical_specialties="cardiology, neurology")
        assert config.medical_specialties == ["cardiology", "neurology"]
        
        # Test with list
        config = MedicalModelConfig(medical_specialties=["cardiology", "neurology"])
        assert config.medical_specialties == ["cardiology", "neurology"]
        
        # Test with None
        config = MedicalModelConfig(medical_specialties=None)
        assert config.medical_specialties == []

    def test_anatomical_regions_validation(self):
        """Test validation of anatomical_regions field."""
        # Test with string
        config = MedicalModelConfig(anatomical_regions="head, chest")
        assert config.anatomical_regions == ["head", "chest"]
        
        # Test with list
        config = MedicalModelConfig(anatomical_regions=["head", "chest"])
        assert config.anatomical_regions == ["head", "chest"]
        
        # Test with None
        config = MedicalModelConfig(anatomical_regions=None)
        assert config.anatomical_regions == []
    
    @patch('os.makedirs')
    def test_model_dir_creation(self, mock_makedirs):
        """Test that model directory is created if it doesn't exist."""
        test_path = "/tmp/test_model_dir"
        config = MedicalModelConfig(model=test_path)
        mock_makedirs.assert_called_once_with(test_path, exist_ok=True)

class TestMedicalModelConfig:
    """Test cases for MedicalModelConfig class."""

    def test_initialization(self, sample_config):
        """Test basic initialization with required parameters."""
        config = MedicalModelConfig(**sample_config)
        
        # Check basic attributes
        assert config.model == sample_config["model"]
        assert config.medical_specialties == sample_config["medical_specialties"]
        assert config.anatomical_regions == sample_config["anatomical_regions"]
        assert config.max_seq_len == sample_config["max_seq_len"]
        assert config.dtype == sample_config["dtype"]

    @pytest.mark.parametrize("specialties,expected", [
        ("cardiology,neurology", ["cardiology", "neurology"]),
        ("cardiology, neurology", ["cardiology", "neurology"]),
        ("cardiology", ["cardiology"]),
        (None, []),
        ("", []),
        (" ", []),
        (["cardiology", "neurology"], ["cardiology", "neurology"]),
        (["cardiology", "  ", "neurology"], ["cardiology", "neurology"]),  # Test whitespace filtering
    ])
    def test_medical_specialties_validation(self, sample_config, specialties, expected):
        """Test validation of medical_specialties field with various inputs."""
        sample_config["medical_specialties"] = specialties
        config = MedicalModelConfig(**sample_config)
        assert config.medical_specialties == expected

    @pytest.mark.parametrize("regions,expected", [
        ("head,chest", ["head", "chest"]),
        ("head, chest", ["head", "chest"]),
        ("head", ["head"]),
        (None, []),
        ("", []),
        (" ", []),
        (["head", "chest"], ["head", "chest"]),
        (["head", "  ", "chest"], ["head", "chest"]),  # Test whitespace filtering
    ])
    def test_anatomical_regions_validation(self, sample_config, regions, expected):
        """Test validation of anatomical_regions field with various inputs."""
        sample_config["anatomical_regions"] = regions
        config = MedicalModelConfig(**sample_config)
        assert config.anatomical_regions == expected

    def test_serialization_roundtrip(self, sample_config, tmp_path):
        """Test serialization and deserialization of MedicalModelConfig."""
        # Create config
        config = MedicalModelConfig(**sample_config)
        
        # Test dict serialization/deserialization
        config_dict = config.to_dict()
        new_config = MedicalModelConfig.from_dict(config_dict)
        assert new_config.to_dict() == config_dict
        
        # Test file serialization/deserialization
        config_path = tmp_path / "config.json"
        config.to_json(config_path)
        loaded_config = MedicalModelConfig.from_json(config_path)
        assert loaded_config.to_dict() == config_dict

    @pytest.mark.cuda
    def test_cuda_specific_tests(self):
        """Tests that require CUDA."""
        # This test will be skipped unless --run-cuda is passed
        assert True  # Replace with actual CUDA tests

    @pytest.mark.slow
    def test_slow_integration(self):
        """Slow integration test."""
        # This test will be skipped unless --run-slow is passed
        assert True  # Replace with actual slow test

    @patch('os.makedirs')
    def test_model_dir_creation(self, mock_makedirs, temp_dir):
        """Test that model directory is created if it doesn't exist."""
        test_path = Path(temp_dir) / "test_model_dir"
        config = MedicalModelConfig(model=test_path)
        mock_makedirs.assert_called_once_with(test_path, exist_ok=True)

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = MedicalModelConfig(model="test-model")
        assert config.medical_specialties == []
        assert config.anatomical_regions == []
        assert config.max_seq_len > 0  # Should have a default value

    def test_invalid_input_handling(self):
        """Test handling of invalid input values."""
        # Test with invalid types
        with pytest.raises(TypeError):
            MedicalModelConfig(model=123)  # model should be string or PathLike
            
        # Test with invalid medical specialties
        with pytest.raises(ValueError):
            MedicalModelConfig(model="test", medical_specialties=[123])  # Should be list of strings

    def test_from_dict_valid(self, sample_config):
        """Test creating config from a valid dictionary."""
        config = MedicalModelConfig.from_dict(sample_config)
        assert isinstance(config, MedicalModelConfig)
        assert config.model_type == "bert"
        assert config.medical_specialties == ["cardiology", "neurology"]
        assert config.anatomical_regions == ["head", "chest"]
        assert Path(config.model).exists()

    def test_from_dict_with_string_specialties(self, sample_config):
        """Test creating config with string input for medical_specialties."""
        config_dict = sample_config.copy()
        config_dict["medical_specialties"] = "cardiology, neurology, radiology"
        config = MedicalModelConfig.from_dict(config_dict)
        assert sorted(config.medical_specialties) == ["cardiology", "neurology", "radiology"]

    def test_from_dict_with_string_anatomical_regions(self, sample_config):
        """Test creating config with string input for anatomical_regions."""
        config_dict = sample_config.copy()
        config_dict["anatomical_regions"] = "head, neck, chest"
        config = MedicalModelConfig.from_dict(config_dict)
        assert sorted(config.anatomical_regions) == ["chest", "head", "neck"]

    def test_invalid_medical_specialties(self, sample_config):
        """Test validation of invalid medical_specialties."""
        config_dict = sample_config.copy()
        config_dict["medical_specialties"] = ["", "  ", "cardiology"]
        with pytest.raises(ValueError):
            MedicalModelConfig.from_dict(config_dict)

    def test_invalid_anatomical_regions(self, sample_config):
        """Test validation of invalid anatomical_regions."""
        config_dict = sample_config.copy()
        config_dict["anatomical_regions"] = [123, "head"]
        with pytest.raises(ValueError):
            MedicalModelConfig.from_dict(config_dict)

    def test_model_path_creation(self, temp_dir):
        """Test that model path is created if it doesn't exist."""
        temp_model_path = Path(temp_dir) / "new_model_dir"
        config_dict = {
            "model": temp_model_path,
            "medical_specialties": [],
            "anatomical_regions": []
        }
        
        # Ensure the directory doesn't exist yet
        assert not temp_model_path.exists()
        
        # Create config - should create the directory
        config = MedicalModelConfig.from_dict(config_dict)
        self.assertTrue(os.path.exists(temp_model_path))
        self.assertEqual(config.model, temp_model_path)

    def test_serialization_roundtrip(self):
        """Test serializing and deserializing the config."""
        config = MedicalModelConfig.from_dict(self.sample_config)
        config_dict = config.to_dict()
        
        # Verify all original fields are present
        for key in self.sample_config:
            self.assertIn(key, config_dict)
            
        # Create new config from serialized dict
        new_config = MedicalModelConfig.from_dict(config_dict)
        self.assertEqual(config.model_type, new_config.model_type)
        self.assertEqual(config.medical_specialties, new_config.medical_specialties)
        self.assertEqual(config.anatomical_regions, new_config.anatomical_regions)


if __name__ == "__main__":
    unittest.main()
