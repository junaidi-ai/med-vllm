from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedModel

from ..sequence import Sequence
from .types import *

if TYPE_CHECKING:
    from medvllm.models.adapters.medical_adapter_base import MedicalModelAdapterBase

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
        # Import here to avoid circular imports
        from medvllm.models.adapters.medical_adapter_base import MedicalModelAdapterBase

        self.adapter: Optional[MedicalModelAdapterBase] = None

    def load_model(self, model_name_or_path: str, **kwargs: Any) -> Any:  # type: ignore[override]
        """Load the model from the registry, hub, or local path.

        This method first tries to load the model from the registry. If the model is not found,
        it falls back to loading from the Hugging Face Hub or local path.

        Args:
            model_name_or_path: Name of the model in the registry, or path/identifier for direct loading.
            **kwargs: Additional arguments to pass to the model loader.

        Returns:
            The loaded model.

        Raises:
            RuntimeError: If the model cannot be loaded.
            ValueError: If the model configuration is invalid.
        """
        from torch.nn import Module
        from transformers import AutoModelForCausalLM

        from medvllm.utils.loader import load_model

        from .registry import registry

        # Set default device and dtype
        device = kwargs.pop("device", self.runner.device)
        dtype = kwargs.pop("torch_dtype", self.runner.dtype)

        try:
            # Try to load from registry first
            model = registry.load_model(
                model_name_or_path,
                device=device,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="auto" if str(device) == "cuda" else None,
                **kwargs,
            )
        except (KeyError, RuntimeError) as e:
            # Fall back to direct loading if not in registry or loading fails
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map="auto" if str(device) == "cuda" else None,
                    **kwargs,
                )

                # If not using device_map, move model to the specified device
                if str(device) != "cuda" or not hasattr(model, "hf_device_map"):
                    model = model.to(device)

            except Exception as inner_e:
                raise RuntimeError(
                    f"Failed to load model '{model_name_or_path}'. "
                    f"Registry error: {str(e)}, Direct load error: {str(inner_e)}"
                ) from inner_e

        # Store the model and its config
        self.model = model  # type: ignore[assignment]
        self._model_config = getattr(model, "config", None)
        if self._model_config is None:
            raise ValueError("Model configuration not found after loading")

        # Set model to evaluation mode if it has eval() method
        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        elif (
            hasattr(model, "module")
            and hasattr(model.module, "eval")
            and callable(model.module.eval)
        ):
            # Handle case where model is wrapped in DataParallel or similar
            model.module.eval()

        # Create medical adapter if enabled
        self._setup_adapter(model, model_name_or_path)

        # Return the model as is, since we've already stored it in self.model
        return model

    def _setup_adapter(
        self, model: Union[Module, PreTrainedModel], model_name_or_path: str
    ) -> None:
        """Set up medical model adapter if enabled.

        Args:
            model: The loaded PyTorch model
            model_name_or_path: Path or name of the model
        """
        config = self.runner.config

        # Check if adapter is enabled
        if not getattr(config, "use_medical_adapter", True):
            return

        try:
            from medvllm.models.adapter_manager import AdapterManager

            # Create adapter
            self.adapter = AdapterManager.create_adapter(
                model=model,
                model_name_or_path=model_name_or_path,
                adapter_type=getattr(config, "adapter_type", None),
                adapter_config=getattr(config, "adapter_config", None),
                hf_config=self._model_config,
            )

            # Setup adapter for inference with tensor parallelism and CUDA optimizations
            use_cuda_graphs = getattr(config, "use_cuda_graphs", False)
            memory_efficient = getattr(config, "memory_efficient", True)
            enable_mixed_precision = getattr(config, "enable_mixed_precision", False)

            self.adapter.setup_for_inference(
                use_cuda_graphs=use_cuda_graphs,
                memory_efficient=memory_efficient,
                enable_mixed_precision=enable_mixed_precision,
            )

            # Move adapter to the correct device
            self.adapter.to(self.runner.device)

            print(
                f"Successfully initialized {self.adapter.model_type} adapter for {model_name_or_path}"
            )

        except Exception as e:
            print(f"Warning: Failed to setup medical adapter: {e}")
            print("Continuing with raw model...")
            self.adapter = None

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

        # Use adapter if available, otherwise use raw model
        if self.adapter is not None:
            # Use the medical adapter for inference
            with torch.no_grad():
                outputs = self.adapter(input_ids, use_cache=True)

            # Extract logits and past_key_values from adapter output
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                past_key_values = getattr(outputs, "past_key_values", None)
            elif isinstance(outputs, tuple):
                logits = outputs[0]
                past_key_values = outputs[1] if len(outputs) > 1 else None
            else:
                logits = outputs
                past_key_values = None
        else:
            # Fallback to raw model
            inputs = self.prepare_inputs(input_ids, positions, is_prefill)

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
    def model_config(self) -> "PretrainedConfigT":
        """Get the model configuration.

        Returns:
            The model configuration.

        Raises:
            RuntimeError: If the model configuration is not loaded.
        """
        if self._model_config is None:
            if self.model is not None:
                self._model_config = getattr(self.model, "config", None)
            if self._model_config is None:
                raise RuntimeError("Model configuration not loaded")
        return self._model_config
