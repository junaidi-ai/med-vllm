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
        
        # Create a dummy model directory with config
        self.model_path = os.path.join(self.temp_dir, "bert-base-uncased")
        os.makedirs(self.model_path, exist_ok=True)
        
        # Create a dummy config file with required fields for a BERT-like model
        self.config_path = os.path.join(self.model_path, "config.json")
        config_data = {
            "model_type": "bert",
            "architectures": ["BertForMaskedLM"],
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
            "classifier_dropout": None,
            "vocab_size": 30522,
            "position_embedding_type": "absolute"
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)
            
        # Create a dummy model file to make the directory look like a valid model
        with open(os.path.join(self.model_path, "pytorch_model.bin"), 'wb') as f:
            f.write(b'dummy model data')
        
        self.sample_config = {
            "model": self.model_path,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 512,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1,
            "enforce_eager": False,
            "kvcache_block_size": 256,
            "num_kvcache_blocks": -1,
            
            # Medical-specific parameters from MedicalModelConfig
            "model_type": "bert",
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

    @patch('transformers.AutoConfig.from_pretrained')
    def test_base_config_compatibility(self, mock_from_pretrained):
        """Test that MedicalModelConfig works with base Config parameters."""
        # Mock the AutoConfig.from_pretrained to return our test config
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_config.max_position_embeddings = 4096
        mock_from_pretrained.return_value = mock_config
        
        # Create with base config parameters only
        base_params = {
            'model': self.model_path,  # Use the path to our test model
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
        
        # Verify default medical parameters are set
        self.assertEqual(config.model_type, "bert")
        self.assertEqual(config.max_medical_seq_length, 512)
        self.assertIsInstance(config.medical_specialties, list)
        self.assertIsInstance(config.anatomical_regions, list)
        self.assertIsInstance(config.enable_uncertainty_estimation, bool)
        
    @patch('transformers.AutoConfig.from_pretrained')
    def test_medical_config_serialization(self, mock_from_pretrained):
        """Test serialization/deserialization maintains all parameters."""
        # Mock the AutoConfig.from_pretrained to return our test config
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_config.max_position_embeddings = 4096
        mock_from_pretrained.return_value = mock_config
        
        # Create config
        config = MedicalModelConfig(**self.sample_config)
        
        # Convert to dict and back
        config_dict = config.to_dict()
        new_config = MedicalModelConfig.from_dict(config_dict)
        
        # Verify all parameters are preserved
        for key, value in config_dict.items():
            if key not in ['model', 'hf_config']:  # Skip model path and hf_config
                self.assertEqual(getattr(new_config, key), value)
            
        # Verify medical-specific parameters
        self.assertEqual(new_config.model_type, "bert")  # Should be 'bert' not 'biobert'
        self.assertEqual(new_config.max_medical_seq_length, self.sample_config["max_medical_seq_length"])
        self.assertEqual(new_config.medical_specialties, self.sample_config["medical_specialties"])
        self.assertEqual(new_config.anatomical_regions, self.sample_config["anatomical_regions"])
        self.assertEqual(new_config.enable_uncertainty_estimation, self.sample_config["enable_uncertainty_estimation"])
            
    @patch('transformers.AutoConfig.from_pretrained')
    def test_json_serialization(self, mock_from_pretrained):
        """Test JSON serialization/deserialization."""
        # Mock the AutoConfig.from_pretrained to return our test config
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_config.max_position_embeddings = 4096
        mock_from_pretrained.return_value = mock_config
        
        # Create a temporary model directory with config.json
        model_dir = os.path.join(self.temp_dir, 'test_model')
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump({'model_type': 'bert'}, f)
        
        # Create config with the test model path
        test_config = self.sample_config.copy()
        test_config['model'] = model_dir
        config = MedicalModelConfig(**test_config)
        
        # Test with file path
        json_path = os.path.join(self.temp_dir, 'medical_config.json')
        config.to_json(json_path)
        
        # Mock the AutoConfig.from_pretrained for loading
        with patch('transformers.AutoConfig.from_pretrained', return_value=mock_config):
            loaded_config = MedicalModelConfig.from_json(json_path)
        
        # Test with string
        json_str = config.to_json()
        with patch('transformers.AutoConfig.from_pretrained', return_value=mock_config):
            loaded_from_str = MedicalModelConfig.from_json(json_str)
        
        # Verify medical-specific parameters
        self.assertEqual(loaded_config.model_type, "bert")
        self.assertEqual(loaded_config.max_medical_seq_length, test_config["max_medical_seq_length"])
        self.assertEqual(loaded_config.medical_specialties, test_config["medical_specialties"])
        self.assertEqual(loaded_config.anatomical_regions, test_config["anatomical_regions"])
        self.assertEqual(loaded_config.enable_uncertainty_estimation, test_config["enable_uncertainty_estimation"])
        
        # Verify the same for string-loaded config
        self.assertEqual(loaded_from_str.model_type, "bert")
        self.assertEqual(loaded_from_str.max_medical_seq_length, test_config["max_medical_seq_length"])
        self.assertEqual(loaded_from_str.medical_specialties, test_config["medical_specialties"])
        self.assertEqual(loaded_from_str.anatomical_regions, test_config["anatomical_regions"])
        self.assertEqual(loaded_from_str.enable_uncertainty_estimation, test_config["enable_uncertainty_estimation"])
    
    @patch('transformers.AutoConfig.from_pretrained')
    def test_backward_compatibility(self, mock_from_pretrained):
        """Test compatibility with base Config usage patterns."""
        # Mock the AutoConfig.from_pretrained to return our test config
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_config.max_position_embeddings = 4096
        mock_from_pretrained.return_value = mock_config
        
        # Create a temporary model directory with config.json
        model_dir = os.path.join(self.temp_dir, 'test_model')
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump({'model_type': 'bert'}, f)
        
        # Create a config with an older version
        old_config = {
            "model": model_dir,  # Use our test model directory
            "config_version": "0.1.0",
            "model_type": "bert",
            "max_medical_seq_length": 256,
            "medical_specialties": ["cardiology"],
            "anatomical_regions": ["head"],
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 512,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1,
            "enforce_eager": False,
            "kvcache_block_size": 256,
            "num_kvcache_blocks": -1
        }
        
        # Load and migrate
        with patch('transformers.AutoConfig.from_pretrained', return_value=mock_config):
            config = MedicalModelConfig.from_dict(old_config)
        
        # Verify migrated values
        self.assertEqual(config.model, model_dir)
        self.assertEqual(config.model_type, "bert")
        self.assertEqual(config.max_medical_seq_length, 256)
        self.assertEqual(config.medical_specialties, ["cardiology"])
        self.assertEqual(config.anatomical_regions, ["head"])
        
        # Test with base parameters only
        base_params = {
            'model': model_dir,
            'max_num_batched_tokens': 8192,
            'max_num_seqs': 256,
            'max_model_len': 512,
            'gpu_memory_utilization': 0.8,
            'tensor_parallel_size': 2,
            'enforce_eager': True,
            'kvcache_block_size': 512,
            'num_kvcache_blocks': 100,
            'model_type': 'bert'
        }
        
        # Should work with just base parameters
        with patch('transformers.AutoConfig.from_pretrained', return_value=mock_config):
            config = MedicalModelConfig(**base_params)
        
        # Verify base parameters are set correctly
        for key, value in base_params.items():
            self.assertEqual(getattr(config, key), value, f"Mismatch for {key}")
        
        # Test with medical parameters
        medical_params = {
            'max_medical_seq_length': 512,
            'medical_specialties': ['cardiology', 'neurology'],
            'anatomical_regions': ['head', 'chest'],
            'enable_uncertainty_estimation': True
        }
        
        # Create config with medical parameters
        full_config = {**base_params, **medical_params}
        with patch('transformers.AutoConfig.from_pretrained', return_value=mock_config):
            config = MedicalModelConfig(**full_config)
        
        # Verify all parameters are set correctly
        for key, value in full_config.items():
            self.assertEqual(getattr(config, key), value, f"Mismatch for {key}")
        
        # Test with from_dict
        with patch('transformers.AutoConfig.from_pretrained', return_value=mock_config):
            config_from_dict = MedicalModelConfig.from_dict(full_config)
        
        for key, value in full_config.items():
            self.assertEqual(getattr(config_from_dict, key), value, f"Mismatch for {key} in from_dict")
        
        # Test with invalid medical parameter values (non-list specialties)
        with self.assertRaises(ValueError):
            with patch('transformers.AutoConfig.from_pretrained', return_value=mock_config):
                MedicalModelConfig(**{**base_params, 'medical_specialties': "not_a_list"})
        
        # Test with invalid tensor_parallel_size
        with self.assertRaises(AssertionError):
            with patch('transformers.AutoConfig.from_pretrained', return_value=mock_config):
                MedicalModelConfig(**{**base_params, 'tensor_parallel_size': -1})
        
        # Test with invalid model path (non-existent directory in temp dir)
        invalid_path = os.path.join(self.temp_dir, 'non_existent_dir')
        with self.assertRaises(ValueError) as cm:
            with patch('transformers.AutoConfig.from_pretrained', side_effect=ValueError("Invalid model path")):
                MedicalModelConfig(**{**base_params, 'model': invalid_path})
        
        # Verify the error message contains information about the invalid path
        error_msg = str(cm.exception).lower()
        self.assertIn('model path does not exist', error_msg.lower())
        
        # Test with invalid tensor_parallel_size (too small)
        with self.assertRaises(AssertionError):
            with patch('transformers.AutoConfig.from_pretrained', return_value=mock_config):
                MedicalModelConfig(**{**base_params, 'tensor_parallel_size': -1})
        
        # Test with invalid tensor_parallel_size (too large)
        with self.assertRaises(AssertionError):
            with patch('transformers.AutoConfig.from_pretrained', return_value=mock_config):
                MedicalModelConfig(**{**base_params, 'tensor_parallel_size': 10})  # Max is 8
        
        print(f"Caught expected ValueError for large tensor_parallel_size")
        


if __name__ == "__main__":
    unittest.main()
