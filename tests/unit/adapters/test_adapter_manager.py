"""Tests for the AdapterManager."""

# Mock transformers to avoid import issues
import sys
import types
from unittest.mock import MagicMock, patch


mock_transformers = types.ModuleType("transformers")
mock_config = types.ModuleType("transformers.configuration_utils")
mock_transformers.AutoConfig = MagicMock()
sys.modules["transformers"] = mock_transformers
sys.modules["transformers.configuration_utils"] = mock_config

from medvllm.models.adapter_manager import AdapterManager


class TestAdapterManager:
    """Test cases for AdapterManager."""

    def test_detect_model_type_biobert(self):
        """Test detection of BioBERT models."""
        test_cases = [
            "dmis-lab/biobert-v1.1",
            "dmis-lab/biobert-base-cased-v1.1",
            "monologg/biobert-v1.1",
            "bio-bert-model",
            "my-bio_bert-custom",
        ]

        for model_name in test_cases:
            detected = AdapterManager.detect_model_type(model_name)
            assert detected == "biobert", f"Failed to detect biobert for {model_name}"

    def test_detect_model_type_clinicalbert(self):
        """Test detection of ClinicalBERT models."""
        test_cases = [
            "emilyalsentzer/Bio_ClinicalBERT",
            "clinical-bert-base",
            "clinicalbert-v1.0",
            "my-clinical_bert-model",
        ]

        for model_name in test_cases:
            detected = AdapterManager.detect_model_type(model_name)
            assert detected == "clinicalbert", f"Failed to detect clinicalbert for {model_name}"

    def test_detect_model_type_pubmedbert(self):
        """Test detection of PubMedBERT models."""
        test_cases = [
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "pubmedbert-base",
            "pubmed-bert-large",
        ]

        for model_name in test_cases:
            detected = AdapterManager.detect_model_type(model_name)
            assert detected == "pubmedbert", f"Failed to detect pubmedbert for {model_name}"

    def test_detect_model_type_bluebert(self):
        """Test detection of BlueBERT models."""
        test_cases = [
            "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
            "bluebert-base",
            "blue-bert-large",
        ]

        for model_name in test_cases:
            detected = AdapterManager.detect_model_type(model_name)
            assert detected == "bluebert", f"Failed to detect bluebert for {model_name}"

    def test_detect_model_type_fallback(self):
        """Test fallback to biobert for unknown models."""
        unknown_models = [
            "unknown-model",
            "bert-base-uncased",
            "gpt2-medium",
            "random-model-name",
        ]

        for model_name in unknown_models:
            detected = AdapterManager.detect_model_type(model_name)
            assert detected == "biobert", f"Failed to fallback to biobert for {model_name}"

    def test_detect_model_type_with_config(self):
        """Test detection using model configuration."""
        mock_config = MagicMock()
        mock_config.model_type = "clinical-bert"

        detected = AdapterManager.detect_model_type("unknown-model", mock_config)
        assert detected == "clinicalbert"

    def test_detect_model_type_with_architecture(self):
        """Test detection using architecture information."""
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_config.architectures = ["BioBertForMaskedLM"]

        detected = AdapterManager.detect_model_type("unknown-model", mock_config)
        assert detected == "biobert"

    def test_get_default_adapter_config_biobert(self):
        """Test default configuration for BioBERT."""
        config = AdapterManager.get_default_adapter_config("biobert")

        assert config["use_kv_cache"] is True
        assert config["use_cuda_graphs"] is False
        assert config["max_batch_size"] == 32
        assert config["max_seq_length"] == 512
        assert config["vocab_size"] == 30522
        assert config["hidden_size"] == 768
        assert config["num_attention_heads"] == 12
        assert config["num_hidden_layers"] == 12

    def test_get_default_adapter_config_clinicalbert(self):
        """Test default configuration for ClinicalBERT."""
        config = AdapterManager.get_default_adapter_config("clinicalbert")

        assert config["use_kv_cache"] is True
        assert config["vocab_size"] == 30522
        assert config["hidden_size"] == 768

    def test_get_default_adapter_config_unknown(self):
        """Test default configuration for unknown adapter type."""
        config = AdapterManager.get_default_adapter_config("unknown")

        # Should still return base config
        assert config["use_kv_cache"] is True
        assert config["max_batch_size"] == 32

    def test_enhance_adapter_config(self):
        """Test configuration enhancement with model parameters."""
        base_config = {"use_kv_cache": True, "max_seq_length": 512}

        mock_model = MagicMock()
        mock_hf_config = MagicMock()
        mock_hf_config.vocab_size = 50000
        mock_hf_config.hidden_size = 1024
        mock_hf_config.num_attention_heads = 16
        mock_hf_config.max_position_embeddings = 1024

        enhanced = AdapterManager._enhance_adapter_config(base_config, mock_model, mock_hf_config)

        assert enhanced["vocab_size"] == 50000
        assert enhanced["hidden_size"] == 1024
        assert enhanced["num_attention_heads"] == 16
        assert enhanced["max_seq_length"] == 512  # Should take minimum
        assert enhanced["use_kv_cache"] is True  # Original config preserved

    @patch("medvllm.models.adapter_manager.create_medical_adapter")
    def test_create_adapter_success(self, mock_create):
        """Test successful adapter creation."""
        mock_model = MagicMock()
        mock_adapter = MagicMock()
        mock_create.return_value = mock_adapter

        adapter = AdapterManager.create_adapter(
            model=mock_model,
            model_name_or_path="dmis-lab/biobert-v1.1",
            adapter_type="biobert",
            adapter_config={"test": "value"},
        )

        assert adapter == mock_adapter
        mock_create.assert_called_once()

    @patch("medvllm.models.adapter_manager.create_medical_adapter")
    def test_create_adapter_fallback(self, mock_create):
        """Test adapter creation with fallback on error."""
        mock_model = MagicMock()
        mock_adapter = MagicMock()

        # First call fails, second succeeds (fallback)
        mock_create.side_effect = [ValueError("Test error"), mock_adapter]

        adapter = AdapterManager.create_adapter(
            model=mock_model, model_name_or_path="unknown-model", adapter_type="unknown"
        )

        assert adapter == mock_adapter
        assert mock_create.call_count == 2
        # Second call should be with biobert fallback
        second_call_args = mock_create.call_args_list[1]
        assert second_call_args[0][1] == "biobert"  # adapter_type argument

    def test_create_adapter_auto_detect(self):
        """Test adapter creation with auto-detection."""
        mock_model = MagicMock()

        with patch("medvllm.models.adapter_manager.create_medical_adapter") as mock_create:
            mock_adapter = MagicMock()
            mock_create.return_value = mock_adapter

            adapter = AdapterManager.create_adapter(
                model=mock_model,
                model_name_or_path="emilyalsentzer/Bio_ClinicalBERT",
                adapter_type=None,  # Auto-detect
            )

            # Should detect clinicalbert and create adapter
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0]
            assert call_args[1] == "clinicalbert"  # adapter_type
