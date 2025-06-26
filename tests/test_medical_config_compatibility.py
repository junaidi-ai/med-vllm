"""
Test script to verify backward compatibility of MedicalModelConfig with base Config.
"""
import os
import sys
import json
import tempfile
import unittest
import importlib
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add parent directory to path to import from nanovllm
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock flash_attn before importing anything that might use it
sys.modules['flash_attn'] = MagicMock()
sys.modules['flash_attn.flash_attn_interface'] = MagicMock()
sys.modules['flash_attn.flash_attn_interface'].flash_attn_varlen_func = lambda *args, **kwargs: None
sys.modules['flash_attn.flash_attn_interface'].flash_attn_with_kvcache = lambda *args, **kwargs: None

# Now import the modules we need to test
from nanovllm.medical import MedicalModelConfig
from nanovllm.config import Config

class TestMedicalConfigCompatibility(unittest.TestCase):    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "test_model")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create a dummy config file with required fields for a BERT-like model
        self.config_path = os.path.join(self.model_dir, "config.json")
        with open(self.config_path, 'w') as f:
            json.dump({
                "model_type": "bert",  # Use a standard model type
                "max_position_embeddings": 4096,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 6,
                "intermediate_size": 3072,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "pad_token_id": 0,
                "type_vocab_size": 2,
                "use_cache": True,
                "classifier_dropout": None
            }, f)
        
        self.sample_config = {
            "model": self.model_dir,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 512,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1,
            "enforce_eager": False,
            "kvcache_block_size": 256,
            "num_kvcache_blocks": -1,
            
            # Medical-specific parameters from MedicalModelConfig
            "model_type": "biobert",
            "max_medical_seq_length": 512,
            "medical_specialties": ["cardiology", "neurology"],
            "anatomical_regions": ["head", "chest", "abdomen"],
            "enable_uncertainty_estimation": True
        }
        
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_base_config_compatibility(self):
        """Test that MedicalModelConfig works with base Config parameters."""
        # Create with base config parameters only
        base_params = {
            'model': self.model_dir,
            'max_num_batched_tokens': 16384,
            'max_num_seqs': 512,
            'max_model_len': 2048,
            'gpu_memory_utilization': 0.8,
            'tensor_parallel_size': 2,
            'enforce_eager': True,
            'kvcache_block_size': 512,
            'num_kvcache_blocks': 100
        }
        config = MedicalModelConfig(**base_params)
        
        # Verify base parameters are set correctly
        self.assertEqual(config.model, base_params['model'])
        self.assertEqual(config.max_num_batched_tokens, base_params['max_num_batched_tokens'])
        self.assertEqual(config.max_num_seqs, base_params['max_num_seqs'])
        self.assertEqual(config.max_model_len, base_params['max_model_len'])
        self.assertEqual(config.gpu_memory_utilization, base_params['gpu_memory_utilization'])
        self.assertEqual(config.tensor_parallel_size, base_params['tensor_parallel_size'])
        self.assertEqual(config.enforce_eager, base_params['enforce_eager'])
        self.assertEqual(config.kvcache_block_size, base_params['kvcache_block_size'])
        self.assertEqual(config.num_kvcache_blocks, base_params['num_kvcache_blocks'])
        
        # Verify medical defaults are set
        self.assertEqual(config.model_type, "biobert")
        self.assertEqual(config.max_medical_seq_length, 512)
        self.assertIsInstance(config.medical_specialties, list)
        self.assertIsInstance(config.anatomical_regions, list)
        self.assertIsInstance(config.enable_uncertainty_estimation, bool)
        
    def test_medical_config_serialization(self):
        """Test serialization/deserialization maintains all parameters."""
        # Create config
        config = MedicalModelConfig(**self.sample_config)
        
        # Convert to dict and back
        config_dict = config.to_dict()
        new_config = MedicalModelConfig.from_dict(config_dict)
        
        # Verify all parameters are preserved
        for key, value in self.sample_config.items():
            self.assertEqual(getattr(new_config, key), value)
            
        # Verify medical-specific parameters
        self.assertEqual(new_config.model_type, self.sample_config["model_type"])
        self.assertEqual(new_config.max_medical_seq_length, self.sample_config["max_medical_seq_length"])
        self.assertEqual(new_config.medical_specialties, self.sample_config["medical_specialties"])
        self.assertEqual(new_config.anatomical_regions, self.sample_config["anatomical_regions"])
        self.assertEqual(new_config.enable_uncertainty_estimation, self.sample_config["enable_uncertainty_estimation"])
            
    def test_json_serialization(self):
        """Test JSON serialization/deserialization."""
        # Create config
        config = MedicalModelConfig(**self.sample_config)
        
        # Test with file path
        json_path = os.path.join(self.temp_dir, 'medical_config.json')
        config.to_json(json_path)
        loaded_config = MedicalModelConfig.from_json(json_path)
        
        # Test with string
        json_str = config.to_json()
        loaded_from_str = MedicalModelConfig.from_json(json_str)
        
        # Verify all parameters are preserved
        for key in self.sample_config:
            self.assertEqual(getattr(loaded_config, key), getattr(config, key), f"Mismatch in {key} for file load")
            self.assertEqual(getattr(loaded_from_str, key), getattr(config, key), f"Mismatch in {key} for string load")
        
        # Verify medical-specific parameters
        self.assertEqual(loaded_config.model_type, config.model_type)
        self.assertEqual(loaded_config.max_medical_seq_length, config.max_medical_seq_length)
        self.assertEqual(loaded_config.medical_specialties, config.medical_specialties)
        self.assertEqual(loaded_config.anatomical_regions, config.anatomical_regions)
        self.assertEqual(loaded_config.enable_uncertainty_estimation, config.enable_uncertainty_estimation)
        
        # Verify the same for string-loaded config
        self.assertEqual(loaded_from_str.model_type, config.model_type)
        self.assertEqual(loaded_from_str.max_medical_seq_length, config.max_medical_seq_length)
        self.assertEqual(loaded_from_str.medical_specialties, config.medical_specialties)
        self.assertEqual(loaded_from_str.anatomical_regions, config.anatomical_regions)
        self.assertEqual(loaded_from_str.enable_uncertainty_estimation, config.enable_uncertainty_estimation)
    
    def test_backward_compatibility(self):
        """Test compatibility with base Config usage patterns."""
        # Ensure the model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Create a minimal config file in the model directory that AutoConfig can load
        config_path = os.path.join(self.model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'model_type': 'bert',
                'hidden_size': 768,
                'num_hidden_layers': 12,
                'num_attention_heads': 12,
                'intermediate_size': 3072,
                'max_position_embeddings': 512,
                'vocab_size': 30522,
                'type_vocab_size': 2,
                'hidden_dropout_prob': 0.1,
                'attention_probs_dropout_prob': 0.1,
                'initializer_range': 0.02,
            }, f)
            
        # Create a dummy vocab file to make it look more like a real model
        vocab_path = os.path.join(self.model_dir, 'vocab.txt')
        with open(vocab_path, 'w') as f:
            f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
            f.write("\n".join([f"[unused{i}]" for i in range(1, 100)]))
        
        # Test direct instantiation with base parameters only
        # Note: max_medical_seq_length defaults to 512 in MedicalModelConfig
        # So we'll set max_model_len to match this default to avoid conflicts
        base_params = {
            'model': self.model_dir,  # Use the test model directory
            'max_num_batched_tokens': 8192,
            'max_num_seqs': 256,
            'max_model_len': 512,  # Match the default max_medical_seq_length
            'gpu_memory_utilization': 0.8,
            'tensor_parallel_size': 2,
            'enforce_eager': True,
            'kvcache_block_size': 512,
            'num_kvcache_blocks': 100,
            'pretrained_model_name_or_path': 'bert-base-uncased',  # Ensure fallback is available
            'model_type': 'clinicalbert'  # Set a valid model type
        }
        
        # Should work with just base parameters
        config = MedicalModelConfig(**base_params)
        
        # Verify base parameters are set correctly
        for key, value in base_params.items():
            if key != 'pretrained_model_name_or_path':  # This might be modified during init
                self.assertEqual(getattr(config, key), value)
        
        # Test with medical parameters
        # Note: max_medical_seq_length should match max_model_len from base_params
        medical_params = {
            'model_type': 'clinicalbert',
            'max_medical_seq_length': 512,  # Match max_model_len from base_params
            'medical_specialties': ['cardiology', 'neurology'],
            'anatomical_regions': ['head', 'chest'],
            'enable_uncertainty_estimation': True
        }
        
        # Create config with medical parameters
        full_config = {**base_params, **medical_params}
        config = MedicalModelConfig(**full_config)
        
        # Verify all parameters are set correctly
        for key, value in full_config.items():
            if key != 'pretrained_model_name_or_path':  # This might be modified during init
                self.assertEqual(getattr(config, key), value)
        
        # Test with from_dict
        config_from_dict = MedicalModelConfig.from_dict(full_config)
        for key, value in full_config.items():
            if key != 'pretrained_model_name_or_path':  # This might be modified during init
                self.assertEqual(getattr(config_from_dict, key), value)
        
        # Test that invalid parameters raise errors
        with self.assertRaises(TypeError):
            MedicalModelConfig(invalid_param="should_fail")
            
        # Test with invalid medical parameter values
        with self.assertRaises(ValueError):
            MedicalModelConfig(**{**base_params, 'model_type': 'invalid_model_type'})
            
        # Test with invalid tensor_parallel_size
        # The base Config class uses assert which raises AssertionError
        # MedicalModelConfig wraps it in a ValueError
        print("\n=== Testing tensor_parallel_size validation ===")
        
        # First, test with a valid tensor_parallel_size to ensure the model loads
        try:
            valid_config = MedicalModelConfig(**{**base_params, 'tensor_parallel_size': 1})
            print(f"Successfully created config with tensor_parallel_size=1")
        except Exception as e:
            self.fail(f"Failed to create config with valid tensor_parallel_size: {e}")
        
        # Now test with invalid value
        # We'll test the validation by checking the error message directly
        with self.assertRaises(ValueError) as cm:
            MedicalModelConfig(**{**base_params, 'tensor_parallel_size': 0})
        
        # Verify the error message contains information about tensor_parallel_size
        error_msg = str(cm.exception).lower()
        print(f"Caught expected ValueError: {error_msg}")
        self.assertIn('tensor_parallel_size', error_msg)
        
        # Also test with a value that's too large
        with self.assertRaises(ValueError) as cm:
            MedicalModelConfig(**{**base_params, 'tensor_parallel_size': 10})  # Max is 8
        
        error_msg = str(cm.exception).lower()
        print(f"Caught expected ValueError for large tensor_parallel_size: {error_msg}")
        self.assertIn('tensor_parallel_size', error_msg)
        


if __name__ == "__main__":
    unittest.main()
