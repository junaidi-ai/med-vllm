from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Union

import torch
import torch.distributed as dist

from .model import ModelManager
from .memory import MemoryManager
from .sampling import SamplingManager
from .types import *

if TYPE_CHECKING:
    pass


class ModelRunner:
    """A class that handles the execution of the model."""

    # Class attributes with type hints
    config: "ConfigT"
    block_size: int
    enforce_eager: bool
    world_size: int
    rank: int
    event: Union["MP_EventT", List["MP_EventT"]]
    model: "Qwen3ForCausalLMT"
    sampler: "SamplerT"
    shm: Optional["SharedMemoryT"] = None
    graphs: "CUDAGraphsT"
    graph_pool: Any  # Type depends on CUDA version, typically graph_pool_handle
    graph_bs: List[int]
    graph_vars: "GraphVarsT"
    kv_cache: Optional["TensorT"] = None
    logits_processor: Optional["LogitsProcessorT"] = None
    past_key_values: "PastKeyValuesT" = None
    _model_config: Optional["PretrainedConfigT"] = None
    _dtype: torch.dtype = torch.float16
    _device: torch.device = torch.device("cuda")

    def __init__(
        self,
        config: "ConfigT",
        world_size: int = 1,
        rank: int = 0,
        distributed_init_method: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the ModelRunner.

        Args:
            config: The model configuration.
            world_size: Number of processes for distributed training.
            rank: Process rank for distributed training.
            distributed_init_method: URL specifying how to initialize the process group.
            device: Device to run the model on. If None, will use CUDA if available.
        """
        # Store basic run configuration first so managers can safely access them
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.enforce_eager = getattr(config, "enforce_eager", False)

        # Set up device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self._device = device

        # Initialize distributed process group if needed
        if world_size > 1:
            if not dist.is_initialized():
                if distributed_init_method is None:
                    raise ValueError(
                        "distributed_init_method must be provided for distributed training"
                    )
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method=distributed_init_method,
                    world_size=world_size,
                    rank=rank,
                )

            # Set device for distributed training
            if torch.cuda.is_available():
                torch.cuda.set_device(rank % torch.cuda.device_count())

        # Initialize model components AFTER config/device are ready
        self.model_manager = ModelManager(self)
        self.memory_manager = MemoryManager(self)
        self.sampling_manager = SamplingManager(self)

        # Initialize model and other components
        self._initialize_model()
        self._initialize_components()

    def _initialize_model(self) -> None:
        """Initialize the model and move it to the appropriate device."""
        raise NotImplementedError("Subclasses must implement _initialize_model")

    def _initialize_components(self) -> None:
        """Initialize other components like samplers, logits processors, etc."""
        raise NotImplementedError("Subclasses must implement _initialize_components")

    def run(
        self,
        seqs: List["SequenceT"],
        is_prefill: bool = False,
    ) -> Optional[List[int]]:
        """Run the model on the given sequences.

        Args:
            seqs: List of sequences to process.
            is_prefill: Whether this is the prefill phase.

        Returns:
            List of sampled token IDs if rank is 0, else None.
        """
        raise NotImplementedError("Subclasses must implement run")

    def _run_model_impl(
        self,
        input_ids: "TensorT",
        positions: "TensorT",
        is_prefill: bool = False,
    ) -> "TensorT":
        """Internal implementation of model execution for CUDA graph capture.

        Args:
            input_ids: Input token IDs.
            positions: Position IDs.
            is_prefill: Whether this is the prefill phase.

        Returns:
            The model output logits.
        """
        if not hasattr(self, "model_manager"):
            raise RuntimeError(
                "Model manager not initialized. Call super().__init__() in your ModelRunner subclass."
            )

        # Delegate to the model manager
        logits, past_key_values = self.model_manager.run_model(input_ids, positions, is_prefill)
        self.past_key_values = past_key_values
        return logits

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, "shm") and self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None

        if hasattr(self, "graphs") and self.graphs is not None:
            self.graphs.clear()

        if hasattr(self, "kv_cache") and self.kv_cache is not None:
            del self.kv_cache
            self.kv_cache = None

        if dist.is_initialized() and dist.is_available():
            dist.destroy_process_group()

    def __del__(self) -> None:
        """Ensure resources are cleaned up when the object is destroyed."""
        self.cleanup()

    @property
    def model_config(self) -> "PretrainedConfigT":
        """Get the model configuration."""
        if self._model_config is None:
            self._model_config = self.model.config
        return self._model_config

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get the model's data type."""
        return self._dtype
