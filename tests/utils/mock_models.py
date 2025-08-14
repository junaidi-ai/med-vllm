"""Mock models for testing Med vLLM.

This module provides mock implementations of medical models for testing purposes.
"""

from typing import Any, Dict, Optional

import torch
from torch import nn


class MockModelOutput(dict):
    """Mock model output class that behaves like a dictionary but allows attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class MockModelMixin:
    """Mixin class providing common functionality for mock models."""

    def __init__(self, config=None):
        self.config = config or {}
        self.device = torch.device("cpu")
        self.training = False

    def to(self, *args, **kwargs) -> "MockModelMixin":
        """Move the model to a device."""
        if args and isinstance(args[0], (str, torch.device)):
            self.device = torch.device(args[0])
        elif "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        return self

    def eval(self) -> "MockModelMixin":
        """Set the model to evaluation mode."""
        self.training = False
        return self

    def train(self, mode: bool = True) -> "MockModelMixin":
        """Set the model to training mode."""
        self.training = mode
        return self

    def state_dict(self):
        """Return an empty state dict."""
        return {}

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load a state dict."""
        pass


class MockMedicalModel(nn.Module, MockModelMixin):
    """Mock MedicalModel for testing purposes."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the mock model."""
        super().__init__()
        MockModelMixin.__init__(self, config)
        self.config = config or {}
        self.config.setdefault("model_type", "medical_llm")
        self.config.setdefault("vocab_size", 30522)
        self.config.setdefault("hidden_size", 768)
        self.config.setdefault("num_hidden_layers", 12)
        self.config.setdefault("num_attention_heads", 12)
        self.config.setdefault("intermediate_size", 3072)
        self.config.setdefault("hidden_dropout_prob", 0.1)
        self.config.setdefault("attention_probs_dropout_prob", 0.1)
        self.config.setdefault("max_position_embeddings", 512)
        self.config.setdefault("type_vocab_size", 2)
        self.config.setdefault("initializer_range", 0.02)

        # Add model-specific attributes
        self.embeddings = nn.Module()
        self.encoder = nn.Module()
        self.pooler = nn.Module()

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, config: Optional[Dict[str, Any]] = None, **kwargs
    ) -> "MockMedicalModel":
        """Create a mock model from a pretrained model."""
        if config is None:
            config = {}

        # Update config with any provided kwargs
        config.update(kwargs)

        # Ensure required config values are set
        config.setdefault("model_name_or_path", model_name_or_path)

        # Create and return the model
        return cls(config=config)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Mock forward pass."""
        batch_size = 1
        seq_length = 10
        hidden_size = self.config.get("hidden_size", 768)

        # Create mock output
        return MockModelOutput(
            {
                "logits": torch.randn(batch_size, seq_length, hidden_size),
                "hidden_states": None,
                "attentions": None,
                "loss": None,
                "last_hidden_state": torch.randn(batch_size, seq_length, hidden_size),
                "pooler_output": torch.randn(batch_size, hidden_size),
            }
        )

    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Mock generation method."""
        # Return tensor of shape (batch_size, max_length) with random token IDs
        batch_size = kwargs.get("batch_size", 1)
        max_length = kwargs.get("max_length", 20)
        return torch.randint(0, self.config["vocab_size"], (batch_size, max_length))


# Make available at module level for easier patching
MedicalModel = MockMedicalModel
