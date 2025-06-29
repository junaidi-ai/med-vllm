"""
Test fixtures for the medical model configuration tests.

This module provides reusable fixtures for testing the medical model configuration system.
"""

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import pytest
import yaml

from medvllm.medical.config import MedicalModelConfig


# Path to the fixtures directory
FIXTURES_DIR = Path(__file__).parent
CONFIGS_DIR = FIXTURES_DIR / "configs"


@pytest.fixture(scope="session")
def sample_config_dict() -> Dict[str, Any]:
    """Return a sample configuration as a dictionary."""
    return {
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


@pytest.fixture
def sample_config(sample_config_dict: Dict[str, Any]) -> MedicalModelConfig:
    """Return a sample MedicalModelConfig instance."""
    return MedicalModelConfig(**sample_config_dict)


@pytest.fixture(scope="session")
def sample_config_path() -> Path:
    """Return the path to the sample JSON config file."""
    return CONFIGS_DIR / "sample_medical_config.json"


@pytest.fixture(scope="session")
def sample_yaml_config_path() -> Path:
    """Return the path to the sample YAML config file."""
    return CONFIGS_DIR / "sample_medical_config.yaml"


@pytest.fixture
def tmp_config_dir(tmp_path: Path, sample_config_dict: Dict[str, Any]) -> Path:
    """Create a temporary directory with a sample config file and return its path."""
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(sample_config_dict, f)
    return tmp_path


@pytest.fixture
def large_config_dict() -> Dict[str, Any]:
    """Return a large configuration for stress testing."""
    base_config = {
        "model_type": "medical_llm",
        "model_name_or_path": "large-medical-model",
        "vocab_size": 50000,
        "hidden_size": 4096,
        "num_hidden_layers": 48,
        "num_attention_heads": 64,
        "intermediate_size": 16384,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 4096,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "use_cache": True,
        "classifier_dropout": 0.1,
        "medical_specialties": [f"specialty_{i}" for i in range(50)],
        "anatomical_regions": [f"region_{i}" for i in range(100)],
        "max_sequence_length": 4096,
    }
    
    # Add some additional nested configuration
    base_config["training_config"] = {
        "learning_rate": 5e-5,
        "num_train_epochs": 10,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "warmup_steps": 1000,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "logging_steps": 500,
        "save_steps": 1000,
        "eval_steps": 1000,
        "save_total_limit": 5,
    }
    
    # Add some custom parameters
    base_config["custom_parameters"] = {
        "enable_gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,
        "fp16": True,
        "fp16_opt_level": "O1",
        "seed": 42,
    }
    
    return base_config


@pytest.fixture
def invalid_config_dict() -> Dict[str, Any]:
    """Return an invalid configuration for testing validation."""
    return {
        "model_type": "invalid_model",
        "hidden_size": -100,  # Invalid value
        "num_hidden_layers": 0,  # Invalid value
        "medical_specialties": [],  # Empty list not allowed
        "anatomical_regions": [123],  # Wrong type
    }


@pytest.fixture(scope="session")
def all_specialties() -> List[str]:
    """Return a list of all allowed medical specialties."""
    return [
        "cardiology", "radiology", "neurology", "oncology", "pediatrics",
        "dermatology", "gastroenterology", "endocrinology", "nephrology",
        "pulmonology", "rheumatology", "urology", "ophthalmology", "otolaryngology",
        "pathology", "psychiatry", "anesthesiology", "emergency_medicine", "family_medicine",
        "internal_medicine", "obstetrics_gynecology", "physical_medicine", "preventive_medicine",
        "radiation_oncology", "surgery"
    ]


@pytest.fixture(scope="session")
def all_anatomical_regions() -> List[str]:
    """Return a list of all allowed anatomical regions."""
    return [
        "head", "neck", "chest", "abdomen", "pelvis", "back", "upper_limb", "lower_limb",
        "brain", "heart", "lungs", "liver", "kidneys", "stomach", "intestines", "bladder",
        "prostate", "ovaries", "uterus", "testes", "pancreas", "spleen", "thyroid", "adrenals"
    ]


@pytest.fixture
def versioned_configs() -> Dict[str, Dict[str, Any]]:
    """Return a dictionary of configurations for different versions."""
    return {
        "0.1.0": {
            "model_type": "medical_llm",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        },
        "1.0.0": {
            "model_type": "medical_llm",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "medical_specialties": ["cardiology", "radiology"],
        },
        "2.0.0": {
            "model_type": "medical_llm",
            "hidden_size": 2048,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "medical_specialties": ["cardiology", "radiology", "neurology"],
            "anatomical_regions": ["head", "chest", "abdomen"],
        },
    }
