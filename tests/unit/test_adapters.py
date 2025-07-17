"""Tests for medical model adapters."""
import pytest
import torch
from unittest.mock import MagicMock, patch

# Mock the transformers module to avoid import errors
import sys
import types

# Create mock transformers module
mock_transformers = types.ModuleType('transformers')
mock_utils = types.ModuleType('transformers.utils')
mock_modeling_utils = types.ModuleType('transformers.modeling_utils')
mock_tokenization_utils = types.ModuleType('transformers.tokenization_utils_base')

# Add necessary attributes to mock modules
mock_transformers.utils = mock_utils
mock_transformers.modeling_utils = mock_modeling_utils
mock_transformers.tokenization_utils_base = mock_tokenization_utils

# Add to sys.modules
sys.modules['transformers'] = mock_transformers
sys.modules['transformers.utils'] = mock_utils
sys.modules['transformers.modeling_utils'] = mock_modeling_utils
sys.modules['transformers.tokenization_utils_base'] = mock_tokenization_utils

# Now import our adapter implementation
from medvllm.models.adapter import (
    MedicalModelAdapter,
    BioBERTAdapter,
    ClinicalBERTAdapter,
    create_medical_adapter
)

# Test implementations
class TestMedicalModelAdapter(MedicalModelAdapter):
    """Concrete implementation of MedicalModelAdapter for testing."""
    def setup_for_inference(self, **kwargs):
        """Set up the model for inference."""
        self.model.eval()
        self.kv_cache = {}
        self.cuda_graphs = None

    def forward(self, input_ids, **kwargs):
        """Forward pass through the model."""
        return self.model(input_ids, **kwargs)


def test_medical_model_adapter_initialization():
    """Test basic initialization of MedicalModelAdapter."""
    mock_model = MagicMock()
    config = {"test_param": "value"}
    
    adapter = TestMedicalModelAdapter(mock_model, config)
    
    assert adapter.model == mock_model
    assert adapter.config == config
    assert adapter.kv_cache is None
    assert adapter.cuda_graphs is None


def test_adapter_setup():
    """Test setting up the adapter for inference."""
    mock_model = MagicMock()
    adapter = TestMedicalModelAdapter(mock_model, {})
    
    # Initial state
    assert adapter.kv_cache is None
    
    # Setup for inference
    adapter.setup_for_inference()
    
    # Verify setup
    mock_model.eval.assert_called_once()
    assert adapter.kv_cache == {}
    assert adapter.cuda_graphs is None


def test_adapter_forward():
    """Test forward pass through the adapter."""
    # Setup mock model
    mock_output = (torch.tensor([[0.1, 0.2, 0.7]]),)
    mock_model = MagicMock(return_value=mock_output)
    
    # Create adapter and setup
    adapter = TestMedicalModelAdapter(mock_model, {})
    adapter.setup_for_inference()
    
    # Test forward pass
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])
    
    output = adapter(input_ids, attention_mask=attention_mask)
    
    # Verify output and calls
    assert torch.equal(output, mock_output[0])
    mock_model.assert_called_once_with(input_ids, attention_mask=attention_mask)


def test_create_medical_adapter():
    """Test the factory function for creating adapters."""
    mock_model = MagicMock()
    
    # Test BioBERT adapter creation
    with patch('medvllm.models.adapter.BioBERTAdapter') as mock_biobert:
        create_medical_adapter(mock_model, "biobert", {})
        mock_biobert.assert_called_once_with(mock_model, {})
    
    # Test ClinicalBERT adapter creation
    with patch('medvllm.models.adapter.ClinicalBERTAdapter') as mock_clinical:
        create_medical_adapter(mock_model, "clinicalbert", {})
        mock_clinical.assert_called_once_with(mock_model, {})
    
    # Test invalid model type
    with pytest.raises(ValueError, match="Unsupported model type"):
        create_medical_adapter(mock_model, "invalid_model", {})
