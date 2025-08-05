"""Base adapter interface for medical language models.

This module provides the abstract base class that all medical model adapters must implement
to be compatible with the Nano vLLM architecture.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch import nn


class MedicalModelAdapter(nn.Module):
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
        self.kv_cache: Optional[Dict[str, torch.Tensor]] = None
        self.cuda_graphs: Optional[Any] = None

        # Tensor parallelism configuration
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)
        self.rank = config.get("rank", 0)
        self.world_size = config.get("world_size", 1)

        # CUDA optimization settings
        self.use_cuda_graphs = config.get("use_cuda_graphs", False)
        self.memory_efficient = config.get("memory_efficient", True)
        self.enable_mixed_precision = config.get("enable_mixed_precision", False)

        # Initialize tensor parallel group if needed
        if self.tensor_parallel_size > 1:
            self._init_tensor_parallel()

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

    def to(self, *args, **kwargs) -> "MedicalModelAdapter":
        """Move the adapter to the specified device and/or dtype.

        This method extends torch.nn.Module.to() to handle the adapter's internal state.

        Args:
            *args: Arguments to pass to the parent to() method.
            **kwargs: Keyword arguments to pass to the parent to() method.

        Returns:
            self: The adapter moved to the specified device/dtype.
        """
        self.model = self.model.to(*args, **kwargs)
        if hasattr(self, "kv_cache") and self.kv_cache is not None:
            if isinstance(self.kv_cache, dict):
                # Handle dictionary-based KV cache
                self.kv_cache = {
                    k: v.to(*args, **kwargs) if v is not None else None
                    for k, v in self.kv_cache.items()
                }
            elif isinstance(self.kv_cache, (tuple, list)):
                # Handle tuple/list-based KV cache (for compatibility with older versions)
                self.kv_cache = tuple(
                    tuple(
                        t.to(*args, **kwargs) if t is not None else None for t in layer
                    )
                    for layer in self.kv_cache
                )
        return self

    def _init_tensor_parallel(self) -> None:
        """Initialize tensor parallelism if not already initialized."""
        if not dist.is_initialized() and self.tensor_parallel_size > 1:
            try:
                # Initialize distributed process group for tensor parallelism
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    rank=self.rank,
                    world_size=self.world_size,
                )
                print(
                    f"Initialized tensor parallelism: rank {self.rank}/{self.world_size}"
                )
            except Exception as e:
                print(f"Warning: Failed to initialize tensor parallelism: {e}")
                self.tensor_parallel_size = 1

    def _shard_tensor(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Shard a tensor across tensor parallel ranks.

        Args:
            tensor: Tensor to shard
            dim: Dimension to shard along

        Returns:
            Sharded tensor for current rank
        """
        if self.tensor_parallel_size <= 1:
            return tensor

        # Calculate shard size and offset
        total_size = tensor.size(dim)
        shard_size = total_size // self.tensor_parallel_size
        start_idx = self.rank * shard_size
        end_idx = start_idx + shard_size

        # Handle last rank getting remainder
        if self.rank == self.tensor_parallel_size - 1:
            end_idx = total_size

        # Shard the tensor
        if dim == 0:
            return tensor[start_idx:end_idx]
        elif dim == 1:
            return tensor[:, start_idx:end_idx]
        elif dim == 2:
            return tensor[:, :, start_idx:end_idx]
        else:
            # Use torch.narrow for higher dimensions
            return torch.narrow(tensor, dim, start_idx, end_idx - start_idx)

    def _all_gather_tensor(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """All-gather a sharded tensor across tensor parallel ranks.

        Args:
            tensor: Sharded tensor to gather
            dim: Dimension that was sharded

        Returns:
            Full tensor gathered from all ranks
        """
        if self.tensor_parallel_size <= 1 or not dist.is_initialized():
            return tensor

        # Gather tensors from all ranks
        tensor_list = [
            torch.zeros_like(tensor) for _ in range(self.tensor_parallel_size)
        ]
        dist.all_gather(tensor_list, tensor)

        # Concatenate along the sharded dimension
        return torch.cat(tensor_list, dim=dim)

    def _optimize_cuda_memory(self) -> None:
        """Optimize CUDA memory usage for medical models."""
        if not torch.cuda.is_available():
            return

        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)

        # Set memory fraction for multi-GPU setups
        if self.tensor_parallel_size > 1:
            memory_fraction = 0.9 / self.tensor_parallel_size
            torch.cuda.set_per_process_memory_fraction(memory_fraction)

        # Enable CUDA graphs compilation cache
        if self.use_cuda_graphs and hasattr(torch.cuda, "memory"):
            # Use public API for CUDA memory management
            torch.cuda.memory._set_allocator_settings("max_split_size_mb:512")

    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training/inference."""
        if self.enable_mixed_precision and torch.cuda.is_available():
            # Enable automatic mixed precision
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

            # Convert model to half precision where appropriate
            if hasattr(self.model, "half"):
                # Only convert non-embedding layers to half
                for name, module in self.model.named_modules():
                    if not isinstance(module, (nn.Embedding, nn.LayerNorm)):
                        module.half()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        stats = {}

        if torch.cuda.is_available():
            stats["cuda_memory_allocated"] = torch.cuda.memory_allocated()
            stats["cuda_memory_reserved"] = torch.cuda.memory_reserved()
            stats["cuda_max_memory_allocated"] = torch.cuda.max_memory_allocated()

        # Model parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        stats["total_parameters"] = total_params
        stats["trainable_parameters"] = trainable_params
        stats["tensor_parallel_size"] = self.tensor_parallel_size
        stats["rank"] = self.rank

        return stats
