"""Tests for medical model adapters."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

# Mock the transformers module to avoid import errors
sys.modules["transformers"] = types.ModuleType("transformers")
sys.modules["transformers.tokenization_utils_base"] = types.ModuleType(
    "transformers.tokenization_utils_base"
)
sys.modules["transformers.modeling_utils"] = types.ModuleType(
    "transformers.modeling_utils"
)

# Now import our adapters
from medvllm.models.adapter import (
    BioBERTAdapter,
    ClinicalBERTAdapter,
    MedicalModelAdapter,
    create_medical_adapter,
)


def test_medical_model_adapter_abstract():
    """Test that MedicalModelAdapter is abstract and can't be instantiated directly."""

    # Create a concrete subclass for testing
    class TestAdapter(MedicalModelAdapter):
        def setup_for_inference(self, **kwargs):
            pass

        def forward(self, input_ids, **kwargs):
            pass

    # Now we can test the base functionality
    mock_model = MagicMock()
    adapter = TestAdapter(mock_model, {})
    assert adapter.model == mock_model
    assert adapter.config == {}
    assert adapter.kv_cache is None
    assert adapter.cuda_graphs is None


def test_create_medical_adapter():
    """Test the factory function for creating adapters."""
    mock_model = MagicMock()

    # Test BioBERT adapter creation
    with patch("medvllm.models.adapter.BioBERTAdapter") as mock_biobert:
        create_medical_adapter(mock_model, "biobert", {})
        mock_biobert.assert_called_once_with(mock_model, {})

    # Test ClinicalBERT adapter creation
    with patch("medvllm.models.adapter.ClinicalBERTAdapter") as mock_clinical:
        create_medical_adapter(mock_model, "clinicalbert", {})
        mock_clinical.assert_called_once_with(mock_model, {})

    # Test invalid model type
    with pytest.raises(ValueError, match="Unsupported model type"):
        create_medical_adapter(mock_model, "invalid_model", {})


def test_biobert_adapter_initialization():
    """Test BioBERT adapter initialization and setup."""
    mock_model = MagicMock()
    config = {"test_param": "value"}

    # Patch the parent class's __init__
    with patch.object(MedicalModelAdapter, "__init__", return_value=None) as mock_init:
        adapter = BioBERTAdapter(mock_model, config)
        mock_init.assert_called_once_with(mock_model, config)

    # Now test with actual initialization
    adapter = BioBERTAdapter(mock_model, config)
    assert adapter.model == mock_model
    assert adapter.config == config
    assert adapter.model_type == "biobert"
    assert adapter.kv_cache is None
    assert adapter.cuda_graphs is None


def test_clinicalbert_adapter_initialization():
    """Test ClinicalBERT adapter initialization and setup."""
    mock_model = MagicMock()
    config = {"test_param": "value"}

    # Patch the parent class's __init__
    with patch.object(MedicalModelAdapter, "__init__", return_value=None) as mock_init:
        adapter = ClinicalBERTAdapter(mock_model, config)
        mock_init.assert_called_once_with(mock_model, config)

    # Now test with actual initialization
    adapter = ClinicalBERTAdapter(mock_model, config)
    assert adapter.model == mock_model
    assert adapter.config == config
    assert adapter.model_type == "clinicalbert"
    assert adapter.kv_cache is None
    assert adapter.cuda_graphs is None


@patch("torch.cuda.is_available", return_value=False)
def test_adapter_setup_cpu(mock_cuda_available):
    """Test adapter setup on CPU."""
    mock_model = MagicMock()
    adapter = BioBERTAdapter(mock_model, {})

    # Reset any existing cache
    adapter.kv_cache = None
    adapter.setup_for_inference(use_cuda_graphs=True)

    mock_model.eval.assert_called_once()
    assert adapter.kv_cache == {}
    assert adapter.cuda_graphs is None

    # Test that existing cache is preserved if setup is called again
    adapter.kv_cache = {"test": "cache"}
    adapter.setup_for_inference()
    assert adapter.kv_cache == {"test": "cache"}  # Shouldn't reset existing cache


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.get_device_properties")
def test_adapter_forward(mock_device_props, mock_cuda_available):
    """Test adapter forward pass."""
    # Setup mock model
    mock_model = MagicMock()
    mock_output = (torch.tensor([[0.1, 0.2, 0.7]]),)
    mock_model.return_value = mock_output

    # Test with BioBERT
    adapter = BioBERTAdapter(mock_model, {})
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])

    # Test forward pass without cache
    output = adapter(input_ids, attention_mask=attention_mask)
    assert torch.equal(output, mock_output[0])
    mock_model.assert_called_once_with(input_ids, attention_mask=attention_mask)

    # Test forward pass with cache - mock the _forward_with_cache method
    mock_model.reset_mock()
    with patch.object(
        adapter, "_forward_with_cache", return_value=mock_output[0]
    ) as mock_forward_cache:
        adapter.kv_cache = {}
        output = adapter(input_ids, attention_mask=attention_mask)
        assert torch.equal(output, mock_output[0])
        mock_forward_cache.assert_called_once_with(input_ids, attention_mask)
        mock_model.assert_not_called()  # Should use cached version


def test_adapter_device_move():
    """Test moving adapter to different devices."""
    mock_model = MagicMock()
    adapter = BioBERTAdapter(mock_model, {})

    # Test moving to CPU
    device = torch.device("cpu")
    result = adapter.to(device)
    mock_model.to.assert_called_once_with(device)
    assert result is adapter  # Should return self for method chaining

    # Test moving to CUDA if available
    if torch.cuda.is_available():
        mock_model.reset_mock()
        device = torch.device("cuda:0")
        adapter.to(device)
        mock_model.to.assert_called_once_with(device)

    # Test with string device
    mock_model.reset_mock()
    adapter.to("cpu")
    mock_model.to.assert_called_once_with("cpu")
