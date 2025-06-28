"""
Tests for MedicalModelConfig functionality including serialization, validation, and compatibility.
"""

import json
import os

# Add parent directory to path to import from nanovllm
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock dependencies
sys.modules["flash_attn"] = MagicMock()
sys.modules["flash_attn.flash_attn_interface"] = MagicMock()
sys.modules["flash_attn.flash_attn_interface"].flash_attn_varlen_func = MagicMock()
sys.modules["flash_attn.flash_attn_interface"].flash_attn_with_kvcache = MagicMock()

from medvllm.medical import MedicalModelConfig


class TestMedicalModelConfig(unittest.TestCase):
    """Test cases for MedicalModelConfig."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a dummy model directory with proper structure
        self.model_dir = os.path.abspath(os.path.join(self.temp_dir, "test_model"))
        os.makedirs(self.model_dir, exist_ok=True)

        # Create a proper model config file
        self.config_path = os.path.join(self.model_dir, "config.json")
        self.model_config = {
            "model_type": "bert",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "vocab_size": 30522,
            "pad_token_id": 0,
        }
        with open(self.config_path, "w") as f:
            json.dump(self.model_config, f)

        # Use default values that match the base Config class defaults
        self.sample_config = {
            "model": self.model_dir,
            "config_version": "1.0.0",
            "max_medical_seq_length": 512,
            "medical_specialties": ["cardiology", "neurology"],
            "anatomical_regions": ["head", "chest"],
            "enable_uncertainty_estimation": True,
            "max_num_batched_tokens": 32768,  # Default from Config class
            "max_num_seqs": 512,  # Default from Config class
            "max_model_len": 4096,  # Default from Config class
            "gpu_memory_utilization": 0.9,  # Default from Config class
            "tensor_parallel_size": 1,  # Default from Config class
            "enforce_eager": False,  # Default from Config class
            "kvcache_block_size": 256,  # Default from Config class
            "num_kvcache_blocks": -1,  # Default from Config class
        }

    def test_config_initialization(self):
        """Test basic configuration initialization."""
        config = MedicalModelConfig(**self.sample_config)
        self.assertEqual(config.max_medical_seq_length, 512)
        self.assertEqual(config.medical_specialties, ["cardiology", "neurology"])
        self.assertTrue(config.enable_uncertainty_estimation)

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        # Create a config with the test model directory
        test_config = self.sample_config.copy()
        test_config["model"] = self.model_dir

        # Create initial config
        config = MedicalModelConfig(**test_config)

        # Convert to dict
        config_dict = config.to_dict()

        # Ensure model path is in the dict (it might be a different key in the serialized form)
        self.assertIn("model", config_dict)
        self.assertEqual(
            os.path.normpath(config_dict["model"]), os.path.normpath(self.model_dir)
        )

        # Create new config from dict
        new_config = MedicalModelConfig.from_dict(config_dict)

        # Verify all fields match
        for key, value in test_config.items():
            if key == "model":
                # Special handling for model path
                self.assertEqual(
                    os.path.normpath(getattr(new_config, key)), os.path.normpath(value)
                )
            elif key == "max_model_len":
                # The model's max_position_embeddings (512) overrides our default (4096)
                self.assertEqual(getattr(new_config, key), 512)
            elif key != "medical_specialties":
                self.assertEqual(getattr(new_config, key), value)

    def test_version_migration(self):
        """Test version migration from older config format."""
        # Create v0.1.0 config
        old_config = {
            "model": self.model_dir,
            "config_version": "0.1.0",
            "max_medical_seq_length": 512,
            "medical_specialties": ["cardiology", "neurology"],
            "anatomical_regions": ["head", "chest"],
            "enable_uncertainty_estimation": True,
        }

        # Save to file to test file-based loading
        config_path = os.path.join(self.temp_dir, "old_config.json")
        with open(config_path, "w") as f:
            json.dump(old_config, f)

        # Test loading from dict
        config = MedicalModelConfig.from_dict(old_config)
        self.assertEqual(config.config_version, "1.0.0")
        self.assertEqual(config.max_medical_seq_length, 512)
        self.assertEqual(config.medical_specialties, ["cardiology", "neurology"])

        # Test loading from file using ConfigSerializer
        from nanovllm.medical.config.serialization import ConfigSerializer

        loaded_config = ConfigSerializer.from_json(MedicalModelConfig, config_path)
        self.assertEqual(loaded_config.config_version, "1.0.0")

    def test_validation(self):
        """Test validation of configuration parameters."""
        # Test invalid sequence length
        with self.assertRaises(ValueError):
            MedicalModelConfig(
                max_medical_seq_length=0,
                **{
                    k: v
                    for k, v in self.sample_config.items()
                    if k != "max_medical_seq_length"
                },
            )

        # Test invalid medical specialties
        with self.assertRaises(ValueError):
            MedicalModelConfig(
                medical_specialties=[123],
                **{
                    k: v
                    for k, v in self.sample_config.items()
                    if k != "medical_specialties"
                },
            )

    def test_file_io(self):
        """Test saving and loading config to/from file."""
        from nanovllm.medical.config.serialization import ConfigSerializer

        # Create a config with the test model directory
        test_config = self.sample_config.copy()
        test_config["model"] = self.model_dir

        # Create config and ensure it's valid
        config = MedicalModelConfig(**test_config)
        config_path = os.path.join(self.temp_dir, "config.json")

        # Save to file using ConfigSerializer
        ConfigSerializer.to_json(config, config_path)

        # Verify the file was created and contains expected data
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            saved_data = json.load(f)
            self.assertIn("model", saved_data)
            self.assertEqual(
                os.path.normpath(saved_data["model"]), os.path.normpath(self.model_dir)
            )

        # Load from file using ConfigSerializer
        loaded_config = ConfigSerializer.from_json(MedicalModelConfig, config_path)

        # Verify all fields match
        for key, value in test_config.items():
            if key == "model":
                # Special handling for model path
                self.assertEqual(
                    os.path.normpath(getattr(loaded_config, key)),
                    os.path.normpath(value),
                )
            elif key == "max_model_len":
                # The model's max_position_embeddings (512) overrides our default (4096)
                self.assertEqual(getattr(loaded_config, key), 512)
            else:
                self.assertEqual(getattr(loaded_config, key), value)


if __name__ == "__main__":
    unittest.main()
