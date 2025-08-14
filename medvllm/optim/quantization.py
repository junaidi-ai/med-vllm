"""Model quantization utilities for medical applications."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    dtype: torch.dtype = torch.qint8
    inplace: bool = False
    exclude_modules: Optional[list] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "dtype": str(self.dtype),
            "inplace": self.inplace,
            "exclude_modules": self.exclude_modules,
        }


def quantize_model(
    model: Union[nn.Module, str], config: Optional[QuantizationConfig] = None, **kwargs
) -> nn.Module:
    """Apply quantization to a model.

    Args:
        model: Model instance or path/name
        config: Quantization configuration
        **kwargs: Override config values

    Returns:
        Quantized model
    """
    if config is None:
        config = QuantizationConfig(**kwargs)

    # Handle string input (model name/path)
    if isinstance(model, str):
        # Try to determine model type from name or config
        if any(name in model.lower() for name in ["gpt", "gpt2", "gptj", "gpt-neo"]):
            model = AutoModelForCausalLM.from_pretrained(model)
        elif any(name in model.lower() for name in ["bert", "roberta", "distilbert"]):
            model = AutoModelForSequenceClassification.from_pretrained(model)
        else:
            model = AutoModel.from_pretrained(model)

    # Get layers to quantize
    exclude_modules = set(config.exclude_modules or [])

    def should_quantize(module_name: str) -> bool:
        """Check if a module should be quantized."""
        if not exclude_modules:
            return True
        return not any(excluded in module_name for excluded in exclude_modules)

    # Prepare model for quantization
    model.eval()

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        # Only quantize Linear and LayerNorm layers by default
        (
            {nn.Linear, nn.LayerNorm}
            if config.exclude_modules is None
            else {m for m in [nn.Linear, nn.LayerNorm] if should_quantize(m.__name__)}
        ),
        dtype=config.dtype,
        inplace=config.inplace,
    )

    return quantized_model


def save_quantized_model(
    model: nn.Module, save_path: str, config: Optional[Dict[str, Any]] = None
) -> None:
    """Save a quantized model with its configuration.

    Args:
        model: Quantized model to save
        save_path: Directory to save the model
        config: Additional configuration to save
    """
    import json
    import os

    os.makedirs(save_path, exist_ok=True)

    # Save model
    model_path = os.path.join(save_path, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)

    # Save config
    config = config or {}
    config["quantization"] = {
        "dtype": str(next(model.parameters()).dtype),
        "quantized": True,
    }

    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def load_quantized_model(
    model_path: str, model_class: Optional[type] = None, **kwargs
) -> nn.Module:
    """Load a quantized model.

    Args:
        model_path: Path to the model directory
        model_class: Model class to use for loading
        **kwargs: Additional arguments for model initialization

    Returns:
        Loaded quantized model
    """
    import json
    import os

    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Initialize model
    if model_class is None:
        if config.get("architectures"):
            model_class_name = config["architectures"][0]
            # This is a simplification - in practice, you'd need to map class names to actual classes
            if "GPT" in model_class_name:
                model_class = AutoModelForCausalLM
            elif "Bert" in model_class_name or "Roberta" in model_class_name:
                model_class = AutoModelForSequenceClassification
            else:
                model_class = AutoModel
        else:
            model_class = AutoModel

    # Import required types
    from typing import TypeVar

    from transformers import AutoModel, PreTrainedModel

    # Define a type variable for the model class
    M = TypeVar("M", bound=PreTrainedModel)

    # Type check for model_class
    if not isinstance(model_class, type):
        raise ValueError("model_class must be a class")

    # Handle the case where model_class is None
    if model_class is None:
        model_class = AutoModel

    # Perform runtime check for from_pretrained method
    from_pretrained = getattr(model_class, "from_pretrained", None)
    if not callable(from_pretrained):
        raise ValueError("model_class must have a callable 'from_pretrained' method")

    # Load the model with the quantized config
    # We've already verified that from_pretrained exists and is callable
    model = model_class.from_pretrained(model_path, **kwargs)  # type: ignore

    # Apply quantization if needed
    if config.get("quantization", {}).get("quantized", False):
        quant_config = QuantizationConfig.from_dict(config["quantization"])
        model = quantize_model(model, quant_config)

    return model
