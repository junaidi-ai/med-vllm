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
    "model": "medical-bert-base",  # Required field
    "model_type": "bert",  # Must be one of the supported model types
    "pretrained_model_name_or_path": "medical-bert-base",
    "max_medical_seq_length": 512,
    "batch_size": 32,
    "enable_uncertainty_estimation": False,
    "uncertainty_threshold": 0.5,
    "cache_ttl": 3600,
    "medical_specialties": ["cardiology", "radiology", "neurology"],
    "anatomical_regions": ["head", "chest"],
    "imaging_modalities": ["xray", "mri"],
    "medical_entity_types": ["DISEASE", "SYMPTOM", "MEDICATION"],
    "ner_confidence_threshold": 0.8,
    "max_entity_span_length": 10,
    "entity_linking": {
        "enabled": False,
        "knowledge_bases": ["umls", "snomed"],
        "confidence_threshold": 0.8
    },
    "document_types": ["clinical_notes", "radiology_reports"],
    "section_headers": ["history", "examination", "assessment"],
    "max_retries": 3,
    "request_timeout": 30,
    "domain_adaptation": False,
    "domain_adaptation_lambda": 0.1,
    "domain_specific_vocab": None,
    "regulatory_compliance": ["hipaa", "gdpr"],
    "config_version": "1.0.0"
}

# Large configuration for stress testing
# Using only valid enum values for enums, and realistic values for other fields
LARGE_CONFIG = {
    **SAMPLE_CONFIG,
    "model": "large-medical-model",
    "pretrained_model_name_or_path": "large-medical-model",
    "max_medical_seq_length": 1024,
    "batch_size": 64,
    "medical_specialties": ["cardiology", "radiology", "neurology", "oncology", "pediatrics"],
    "anatomical_regions": ["head", "chest", "abdomen", "pelvis", "upper_limb"],
    "imaging_modalities": ["xray", "ct", "mri", "ultrasound", "pet"],
    "medical_entity_types": ["DISEASE", "SYMPTOM", "MEDICATION", "TREATMENT", "LAB_TEST", "ANATOMY"],
    "document_types": ["clinical_notes", "radiology_reports", "discharge_summaries", "progress_notes"],
    "section_headers": ["history", "examination", "assessment", "plan", "medications"],
    "entity_linking": {
        "enabled": True,
        "knowledge_bases": ["umls", "snomed", "icd10", "rxnorm"],
        "confidence_threshold": 0.8
    },
    "domain_specific_vocab": {
        "cardiology": ["echocardiogram", "stent", "angioplasty", "tachycardia", "myocardial"],
        "oncology": ["chemotherapy", "biopsy", "metastasis", "carcinoma", "malignancy"],
        "neurology": ["migraine", "seizure", "neuropathy", "cerebral", "cognitive"]
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
        benchmark(config.to_json)
    
    @pytest.mark.performance
    def test_config_to_json_large(self, benchmark) -> None:
        """Benchmark conversion of large config to JSON string."""
        config = MedicalModelConfig(**LARGE_CONFIG)
        benchmark(config.to_json)


@pytest.fixture(scope="module")
def large_config_file(tmp_path_factory) -> Path:
    """Create a temporary directory with a large configuration for testing."""
    # Create a temporary directory
    temp_dir = tmp_path_factory.mktemp("large_config")
    
    # Create a config with the large configuration
    config = MedicalModelConfig(**LARGE_CONFIG)
    
    # Save the config to the temporary directory
    config_path = temp_dir / "config.json"
    config.to_json(config_path)
    
    return config_path


def test_config_loading_from_large_file(benchmark, large_config_file: Path) -> None:
    """Benchmark loading configuration from a large config file."""
    def load_config() -> None:
        # Load the config from the file using from_dict
        with open(large_config_file, "r") as f:
            config_dict = json.load(f)
        MedicalModelConfig.from_dict(config_dict)
    
    benchmark(load_config)
