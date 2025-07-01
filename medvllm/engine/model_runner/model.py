from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from ..sequence import Sequence
from .types import *

if TYPE_CHECKING:
    from .base import ModelRunner


class ModelManager:
    """Manages model loading and execution."""

    def __init__(self, runner: "ModelRunner") -> None:
        """Initialize the model manager.

        Args:
            runner: The parent ModelRunner instance.
        """
        self.runner = runner
        self.model: Optional[Module] = None
        self._model_config: Optional[PretrainedConfigT] = None

    def load_model(self, model_name_or_path: str, **kwargs: Any) -> Module:
        """Load the model from a checkpoint or hub.

        Args:
            model_name_or_path: Path to the model or model name.
            **kwargs: Additional arguments to pass to the model loader.

        Returns:
            The loaded model.
        """
        from medvllm.utils.loader import load_model

        # Set default device and dtype
        device = kwargs.pop("device", self.runner.device)
        dtype = kwargs.pop("torch_dtype", self.runner.dtype)

        # Load the model using the actual function signature from loader.py
        model = load_model(
            model=model_name_or_path,  # First argument is the model
            path=model_name_or_path,  # Second argument is the path
        )

        # Move model to the specified device
        model = model.to(device)

        # Store the model and its config
        self.model = model
        self._model_config = getattr(model, "config", None)

        # Set model to evaluation mode
        model.eval()

        return model

    def prepare_inputs(
        self,
        input_ids: TensorT,
        positions: TensorT,
        is_prefill: bool = False,
    ) -> Dict[str, Any]:
        """Prepare model inputs.

        Args:
            input_ids: Input token IDs.
            positions: Position IDs.
            is_prefill: Whether this is the prefill phase.

        Returns:
            Dictionary of model inputs.
        """
        inputs = {
            "input_ids": input_ids,
            "position_ids": positions,
            "use_cache": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }

        # Add past_key_values for decode phase
        if not is_prefill and self.runner.past_key_values is not None:
            inputs["past_key_values"] = self.runner.past_key_values

        return inputs

    def run_model(
        self,
        input_ids: TensorT,
        positions: TensorT,
        is_prefill: bool = False,
    ) -> Tuple[TensorT, Optional[PastKeyValuesT]]:
        """Run the model on the given inputs.

        Args:
            input_ids: Input token IDs.
            positions: Position IDs.
            is_prefill: Whether this is the prefill phase.

        Returns:
            Tuple of (logits, past_key_values).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Prepare inputs
        inputs = self.prepare_inputs(input_ids, positions, is_prefill)

        # Run the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract logits and past_key_values
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        past_key_values = getattr(outputs, "past_key_values", None)

        return logits, past_key_values

    def update_past_key_values(self, past_key_values: PastKeyValuesT) -> None:
        """Update the past key-values cache.

        Args:
            past_key_values: The new past key-values.
        """
        self.runner.past_key_values = past_key_values

    @property
    def model_config(self) -> PretrainedConfigT:
        """Get the model configuration."""
        if self._model_config is None and self.model is not None:
            self._model_config = getattr(self.model, "config", None)
        return self._model_config
