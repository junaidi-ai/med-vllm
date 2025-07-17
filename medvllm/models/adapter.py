"""Adapter interface for medical language models.

This module provides a flexible adapter interface to integrate various medical language models
with the Nano vLLM architecture, ensuring consistent behavior and optimization.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


class MedicalModelAdapter(ABC, nn.Module):
    """Abstract base class for medical model adapters.
    
    This class defines the interface that all medical model adapters must implement
    to be compatible with the Nano vLLM architecture.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize the adapter with a model and configuration.
        
        Args:
            model: The underlying model to adapt
            config: Configuration dictionary for the adapter
        """
        super().__init__()
        self.model = model
        self.config = config
        self.kv_cache = None
        self.cuda_graphs = None
    
    @abstractmethod
    def setup_for_inference(self, **kwargs) -> None:
        """Prepare the model for inference with optimizations.
        
        This should be called before any inference to set up CUDA graphs,
        KV cache, and other optimizations.
        """
        pass
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments for the forward pass
            
        Returns:
            Model outputs
        """
        pass
    
    def reset_cache(self) -> None:
        """Reset the KV cache if it exists."""
        if self.kv_cache is not None:
            self.kv_cache = None
    
    def to(self, device: torch.device) -> 'MedicalModelAdapter':
        """Move the model to the specified device."""
        self.model = self.model.to(device)
        return self


class BioBERTAdapter(MedicalModelAdapter):
    """Adapter for BioBERT models."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.model_type = "biobert"
    
    def setup_for_inference(self, use_cuda_graphs: bool = False, **kwargs) -> None:
        """Set up BioBERT for inference with optimizations."""
        self.model.eval()
        
        # Initialize KV cache if not already done
        if self.kv_cache is None:
            self.kv_cache = self._initialize_kv_cache()
        
        # Set up CUDA graphs if requested and available
        if use_cuda_graphs and torch.cuda.is_available():
            self._setup_cuda_graphs()
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for BioBERT."""
        attention_mask = kwargs.get('attention_mask')
        
        # Use KV cache if available
        if self.kv_cache is not None:
            outputs = self._forward_with_cache(input_ids, attention_mask, **kwargs)
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
            
        return outputs[0] if isinstance(outputs, tuple) else outputs
    
    def _initialize_kv_cache(self) -> Dict[str, torch.Tensor]:
        """Initialize KV cache for faster inference."""
        # Implementation depends on the specific model architecture
        return {}
    
    def _forward_with_cache(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass with KV cache."""
        # Implementation for cached forward pass
        pass
    
    def _setup_cuda_graphs(self) -> None:
        """Set up CUDA graphs for faster inference."""
        if not torch.cuda.is_available():
            return
        # Implementation for CUDA graphs
        pass


class ClinicalBERTAdapter(MedicalModelAdapter):
    """Adapter for ClinicalBERT models."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.model_type = "clinicalbert"
    
    def setup_for_inference(self, use_cuda_graphs: bool = False, **kwargs) -> None:
        """Set up ClinicalBERT for inference with optimizations."""
        self.model.eval()
        
        # Initialize KV cache if not already done
        if self.kv_cache is None:
            self.kv_cache = self._initialize_kv_cache()
        
        # Set up CUDA graphs if requested and available
        if use_cuda_graphs and torch.cuda.is_available():
            self._setup_cuda_graphs()
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for ClinicalBERT."""
        attention_mask = kwargs.get('attention_mask')
        
        # Use KV cache if available
        if self.kv_cache is not None:
            outputs = self._forward_with_cache(input_ids, attention_mask, **kwargs)
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
            
        return outputs[0] if isinstance(outputs, tuple) else outputs
    
    def _initialize_kv_cache(self) -> Dict[str, torch.Tensor]:
        """Initialize KV cache for faster inference."""
        # Implementation depends on the specific model architecture
        return {}
    
    def _forward_with_cache(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass with KV cache."""
        # Implementation for cached forward pass
        pass
    
    def _setup_cuda_graphs(self) -> None:
        """Set up CUDA graphs for faster inference."""
        if not torch.cuda.is_available():
            return
        # Implementation for CUDA graphs
        pass


def create_medical_adapter(
    model: nn.Module, 
    model_type: str, 
    config: Dict[str, Any]
) -> MedicalModelAdapter:
    """Factory function to create the appropriate adapter for a medical model.
    
    Args:
        model: The model to adapt
        model_type: Type of the model ('biobert' or 'clinicalbert')
        config: Configuration for the adapter
        
    Returns:
        An instance of the appropriate adapter
    """
    model_type = model_type.lower()
    if model_type == 'biobert':
        return BioBERTAdapter(model, config)
    elif model_type == 'clinicalbert':
        return ClinicalBERTAdapter(model, config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


__all__ = [
    'MedicalModelAdapter',
    'BioBERTAdapter',
    'ClinicalBERTAdapter',
    'create_medical_adapter'
]
