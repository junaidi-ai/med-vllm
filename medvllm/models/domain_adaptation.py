"""Domain adaptation module for medical language models."""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

class DomainAdapter(nn.Module):
    """Adapter for domain adaptation of medical language models."""
    
    def __init__(self, model: nn.Module, **kwargs):
        """Initialize the domain adapter.
        
        Args:
            model: The base model to adapt.
            **kwargs: Additional arguments for the adapter.
        """
        super().__init__()
        self.model = model
        
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the adapter.
        
        Args:
            *args: Positional arguments to pass to the model.
            **kwargs: Keyword arguments to pass to the model.
            
        Returns:
            The output of the model.
        """
        return self.model(*args, **kwargs)


def adapt_model_for_domain(model: nn.Module, **kwargs) -> nn.Module:
    """Adapt a model for a specific domain.
    
    Args:
        model: The model to adapt.
        **kwargs: Additional arguments for the adapter.
        
    Returns:
        The adapted model.
    """
    return DomainAdapter(model, **kwargs)
