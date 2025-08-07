"""Mock adapters for testing Med vLLM.

This module provides mock implementations of medical model adapters for testing purposes.
"""
from typing import Any, Dict, Optional

import torch
from torch import nn


class MockAdapterBase(nn.Module):
    """Base class for mock adapters."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = kwargs.get('config', {})
        self.model = nn.Module()
        self.kv_cache = None
        self.cuda_graphs = None
        self.tensor_parallel_size = 1
        self.rank = 0
        self.world_size = 1
        self.use_cuda_graphs = False
        self.memory_efficient = True
        self.enable_mixed_precision = False
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Create a mock adapter from a pretrained model."""
        return cls(config={"model_type": cls.__name__.lower(), **kwargs})
    
    def setup_for_inference(self, **kwargs):
        """Set up the adapter for inference."""
        pass
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Mock forward pass."""
        return torch.randn(input_ids.shape[0], 10)  # Random logits
    
    def to(self, *args, **kwargs):
        """Move the adapter to a device."""
        return self
    
    def eval(self):
        """Set the adapter to evaluation mode."""
        return self


class MockBioBERTAdapter(MockAdapterBase):
    """Mock BioBERT adapter for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config["model_type"] = "biobert"
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Create a mock BioBERT adapter."""
        instance = super().from_pretrained(model_name_or_path, **kwargs)
        instance.config["model_type"] = "biobert"
        return instance


class MockClinicalBERTAdapter(MockAdapterBase):
    """Mock ClinicalBERT adapter for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config["model_type"] = "clinical_bert"
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Create a mock ClinicalBERT adapter."""
        return cls(config={"model_type": "clinical_bert", **kwargs})


# Create aliases for the actual classes used in the codebase
class BioBERTAdapter(MockBioBERTAdapter):
    """Alias for MockBioBERTAdapter for testing."""
    pass


class ClinicalBERTAdapter(MockClinicalBERTAdapter):
    """Alias for MockClinicalBERTAdapter for testing."""
    pass


class MedicalModelAdapterBase(MockAdapterBase):
    """Alias for MockAdapterBase for testing."""
    pass


__all__ = [
    "MockAdapterBase",
    "MockBioBERTAdapter",
    "MockClinicalBERTAdapter",
    "BioBERTAdapter",
    "ClinicalBERTAdapter",
    "MedicalModelAdapterBase"
]
