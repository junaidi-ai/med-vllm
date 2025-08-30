from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import logging

import torch
from torch.nn import Module
from transformers import PreTrainedModel

from .types import *
from medvllm.utils.profiler import get_profiler

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

        self.adapter: Optional[MedicalModelAdapterBase] = None

        # Module-level logger
        self._logger = logging.getLogger(__name__)

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
        from transformers import AutoModelForCausalLM

        from .registry import registry

        # Set default device and dtype
        device = kwargs.pop("device", self.runner.device)
        dtype = kwargs.pop("torch_dtype", self.runner.dtype)

        # Quantization preferences from Config
        q_bits = getattr(self.runner.config, "quantization_bits", None)
        q_method = (getattr(self.runner.config, "quantization_method", None) or "").lower()

        # Helper: attempt bitsandbytes-aware loading when requested
        def _load_with_bnb(bits: int, method: str) -> Optional[PreTrainedModel]:
            try:
                # Prefer direct HF loading with bnb flags
                load_kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                }
                if bits == 8:
                    load_kwargs["load_in_8bit"] = True
                elif bits == 4:
                    load_kwargs["load_in_4bit"] = True
                    # Default to nf4 if method mentions nf4
                    if "nf4" in method:
                        load_kwargs["bnb_4bit_quant_type"] = "nf4"
                # device mapping for quantized load
                load_kwargs["device_map"] = "auto"

                model_ = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    **load_kwargs,
                )
                return model_
            except Exception:
                return None

        # If bitsandbytes quantization requested, try that path first (bypass registry)
        if q_bits in (4, 8) and ("bnb" in q_method):
            model = _load_with_bnb(int(q_bits), q_method)
            if model is None:
                # Fall back to regular paths if bnb load failed
                pass

        # (moved) backend/runtime optimizations will be applied after model is loaded and optional quantization

        if 'model' not in locals() or model is None:  # type: ignore[name-defined]
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
                    # If bnb was requested but earlier path failed, try once more here
                    if q_bits in (4, 8) and ("bnb" in q_method):
                        model = _load_with_bnb(int(q_bits), q_method)
                    if not model:
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

        # Apply post-load dynamic quantization on CPU if requested and not bnb
        if q_bits == 8 and (q_method in ("", None, "dynamic", "torch", "cpu")):
            try:
                # Dynamic quantization works on CPU modules
                from medvllm.optim.quantization import quantize_model, QuantizationConfig

                if str(device) != "cpu":
                    # create a cpu copy for quantized eval
                    model_cpu = model.to("cpu")
                else:
                    model_cpu = model
                qcfg = QuantizationConfig(dtype=torch.qint8, inplace=False)
                model = quantize_model(model_cpu, qcfg)
                # keep on CPU after dynamic quantization
            except Exception:
                # Best-effort; continue with original model
                pass

        # Apply backend/runtime optimizations based on Config (after model is ready)
        try:
            cfg = self.runner.config
            # Torch backends and matmul precision
            allow_tf32 = bool(getattr(cfg, "allow_tf32", False))
            mm_prec = getattr(cfg, "torch_matmul_precision", None)
            cudnn_bench = getattr(cfg, "cudnn_benchmark", None)
            compile_enabled = bool(getattr(cfg, "enable_torch_compile", False))
            compile_mode = getattr(cfg, "torch_compile_mode", None) or "default"
            if torch.cuda.is_available():
                try:
                    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
                except Exception:
                    pass
                if isinstance(cudnn_bench, bool):
                    try:
                        torch.backends.cudnn.benchmark = cudnn_bench
                    except Exception:
                        pass
            if isinstance(mm_prec, str):
                try:
                    torch.set_float32_matmul_precision(mm_prec)  # type: ignore[attr-defined]
                except Exception:
                    pass

            # FlashAttention (best-effort; always attempt to forward config and call enabler)
            enable_fa = getattr(cfg, "enable_flash_attention", None)
            if enable_fa is True and model is not None:
                device_str = str(self.runner.device)
                # Warn if CUDA is unavailable but still attempt a best-effort enable (no-op in CPU tests)
                if not torch.cuda.is_available() or device_str != "cuda":
                    self._logger.warning(
                        "Flash Attention requested but CUDA is unavailable or device is not 'cuda'. Falling back."
                    )
                try:
                    from medvllm.optim.flash_attention import (
                        FlashAttentionConfig,
                        enable_flash_attention,
                    )

                    fa_kwargs = getattr(cfg, "flash_attention_config", None) or {}
                    fa_cfg = (
                        FlashAttentionConfig.from_dict(fa_kwargs)
                        if isinstance(fa_kwargs, dict)
                        else FlashAttentionConfig()
                    )
                    # Always call enabler; on CPU this should be a no-op and satisfies integration tests
                    model = enable_flash_attention(model, config=fa_cfg)
                except Exception as e:
                    # If FA not available or fails, warn and continue
                    self._logger.warning(
                        f"Failed to enable Flash Attention; continuing without it. Reason: {e}"
                    )

            # Gradient checkpointing (best-effort; eval stays set for inference)
            if bool(getattr(cfg, "grad_checkpointing", False)) and model is not None:
                try:
                    if hasattr(model, "gradient_checkpointing_enable"):
                        model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
                    elif hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()  # type: ignore[attr-defined]
                except Exception:
                    pass

            # Optional torch.compile for op fusion via Inductor (best-effort)
            if compile_enabled and model is not None:
                try:
                    model = torch.compile(model, mode=compile_mode)  # type: ignore[attr-defined]
                except Exception:
                    pass

            # Persist possibly patched model
            self.model = model  # type: ignore[assignment]
        except Exception:
            # Non-fatal: continue with loaded model as-is
            pass

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

            # Merge runtime flags into adapter_config so adapters can consume them
            base_adapter_cfg = getattr(config, "adapter_config", None) or {}
            adapter_cfg: Dict[str, Any] = dict(base_adapter_cfg)
            # Pass through relevant flags if set
            for key in (
                "attention_impl",
                "enable_mixed_precision",
                "mixed_precision_dtype",
                "enable_profiling",
                "profiler_device",
                # Memory pooling flags for adapter awareness
                "enable_memory_pooling",
                "pool_max_bytes",
                "pool_device",
            ):
                val = getattr(config, key, None)
                if val is not None:
                    adapter_cfg[key] = val

            # Create adapter
            self.adapter = AdapterManager.create_adapter(
                model=model,
                model_name_or_path=model_name_or_path,
                adapter_type=getattr(config, "adapter_type", None),
                adapter_config=adapter_cfg,
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

        # Optional mixed precision autocast
        mp_enabled = bool(getattr(self.runner.config, "enable_mixed_precision", False))
        mp_dtype_str = getattr(self.runner.config, "mixed_precision_dtype", "fp16")
        device_type = (
            "cuda" if torch.cuda.is_available() and str(self.runner.device) == "cuda" else "cpu"
        )
        mp_dtype = torch.float16 if str(mp_dtype_str).lower() == "fp16" else torch.bfloat16

        # Optional profiling (unified)
        prof_enabled = bool(getattr(self.runner.config, "enable_profiling", False))
        prof_device_pref = getattr(self.runner.config, "profiler_device", "auto")
        prof_device = (
            "cuda"
            if (prof_device_pref == "auto" and device_type == "cuda")
            else (prof_device_pref or "cpu")
        )
        emit_trace = bool(getattr(self.runner.config, "emit_trace", False))
        trace_dir = getattr(self.runner.config, "trace_dir", None)
        profiler = None
        prof_cm = None
        if prof_enabled:
            try:
                profiler = get_profiler(
                    device=prof_device, emit_trace=emit_trace, trace_dir=trace_dir
                )
                prof_cm = profiler.profile()
                prof_cm.__enter__()
            except Exception:
                profiler = None
                prof_cm = None

        try:
            # Use adapter if available, otherwise use raw model
            if self.adapter is not None:
                # Use the medical adapter for inference
                with torch.no_grad():
                    if mp_enabled and device_type == "cuda":
                        with torch.autocast(device_type=device_type, dtype=mp_dtype):
                            outputs = self.adapter(input_ids, use_cache=True)
                    else:
                        outputs = self.adapter(input_ids, use_cache=True)
            else:
                # Fallback to raw model
                inputs = self.prepare_inputs(input_ids, positions, is_prefill)

                with torch.no_grad():
                    if mp_enabled and device_type == "cuda":
                        with torch.autocast(device_type=device_type, dtype=mp_dtype):
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)

            # Extract logits and past_key_values from outputs
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                past_key_values = getattr(outputs, "past_key_values", None)
            elif isinstance(outputs, tuple):
                logits = outputs[0]
                past_key_values = outputs[1] if len(outputs) > 1 else None
            else:
                logits = outputs
                past_key_values = None

        finally:
            # Exit profiler if entered
            if prof_cm is not None:
                try:
                    prof_cm.__exit__(None, None, None)
                except Exception:
                    pass
            # Attach profiling results for downstream consumers if needed
            # Start with profiler results if available; otherwise create a fresh dict
            profile_dict: Dict[str, Any] = {}
            if profiler is not None and hasattr(profiler, "results"):
                try:
                    profile_dict = dict(getattr(profiler, "results", {}) or {})
                except Exception:
                    profile_dict = {}
            # Attach a snapshot of memory pool stats
            try:
                mm = getattr(self.runner, "memory_manager", None)
                if mm is not None and hasattr(mm, "pool_stats"):
                    # Shallow copy to avoid external mutation
                    profile_dict["memory_pool"] = dict(mm.pool_stats)
            except Exception:
                pass
            # Persist
            try:
                setattr(self.runner, "last_profile", profile_dict)
            except Exception:
                pass

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
