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

    # Apply dynamic quantization if available; otherwise, no-op
    qapi = getattr(torch, "quantization", None)
    qfunc = getattr(qapi, "quantize_dynamic", None) if qapi is not None else None
    if callable(qfunc):
        quantized_model = qfunc(  # type: ignore[misc]
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
    # Fallback: return original model unmodified
    return model


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


# --- Bitsandbytes helpers (best-effort wrappers) ---
def bnb_load_quantized(
    model_name_or_path: str,
    bits: int = 8,
    method: str = "bnb-8bit",
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = True,
    **kwargs: Any,
):
    """Load a model in 8-bit or 4-bit using transformers + bitsandbytes flags.

    Args:
        model_name_or_path: HF repo id or local path.
        bits: 8 or 4.
        method: "bnb-8bit" or "bnb-nf4" (nf4 for 4-bit).
        device_map: device map to use (e.g., "auto").
        trust_remote_code: pass-through for HF loading.
        **kwargs: forwarded to from_pretrained.

    Returns:
        Loaded model (PreTrainedModel).
    """
    load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if bits == 8:
        load_kwargs["load_in_8bit"] = True
    elif bits == 4:
        load_kwargs["load_in_4bit"] = True
        if "nf4" in (method or "").lower():
            load_kwargs["bnb_4bit_quant_type"] = "nf4"
    else:
        raise ValueError(f"Unsupported bits for bnb_load_quantized: {bits}")

    if device_map is not None:
        load_kwargs["device_map"] = device_map

    load_kwargs.update(kwargs)
    # Default to CausalLM as common path; callers can override class via kwargs if needed
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)


def bnb_save_stub(save_dir: str, info: Optional[Dict[str, Any]] = None) -> None:
    """Best-effort stub to save metadata for a bnb-quantized model.

    Actual state dict saving for 4-bit/8-bit is environment/format specific.
    This stub records metadata so users can re-load via bnb_load_quantized.
    """
    import os
    import json

    os.makedirs(save_dir, exist_ok=True)
    meta = {"note": "To reload, call bnb_load_quantized with model_name_or_path and method/bits."}
    if info:
        meta.update(info)
    with open(os.path.join(save_dir, "bnb_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def bnb_offline_hint() -> str:
    return (
        "Saving true 4/8-bit bnb weights offline is not standardized across models. "
        "Prefer loading with from_pretrained(load_in_4bit/8bit) and using device_map='auto'."
    )
