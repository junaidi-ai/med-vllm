from __future__ import annotations

import ctypes
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor

from .types import *

if TYPE_CHECKING:
    from .base import ModelRunner


class MemoryManager:
    """Manages shared memory and KV cache for the model."""

    def __init__(self, runner: "ModelRunner") -> None:
        """Initialize the memory manager.

        Args:
            runner: The parent ModelRunner instance.
        """
        self.runner = runner
        self.shm: Optional[SharedMemoryT] = None
        self.kv_cache: Optional[TensorT] = None

    def allocate_kv_cache(self, gpu_memory_utilization: float = 0.9) -> None:
        """Allocate GPU memory for the key-value cache.

        Args:
            gpu_memory_utilization: Fraction of GPU memory to use for KV cache.
        """
        if not (0 < gpu_memory_utilization <= 1.0):
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}"
            )

        if not torch.cuda.is_available():
            return

        # Get GPU memory info
        total_mem = torch.cuda.get_device_properties(self.runner.device).total_memory
        reserved_mem = torch.cuda.memory_reserved(self.runner.device)
        free_mem = total_mem - reserved_mem

        # Calculate cache size
        cache_bytes = int(free_mem * gpu_memory_utilization)

        # Get model config
        config = self.runner.model_config

        # Calculate shape for KV cache
        num_layers = getattr(config, "num_hidden_layers", 0)
        num_heads = getattr(config, "num_attention_heads", 0)
        head_dim = (
            getattr(config, "head_dim", 0)
            or getattr(config, "hidden_size", 0) // num_heads
        )

        if num_layers == 0 or num_heads == 0 or head_dim == 0:
            raise ValueError(
                "Could not determine model architecture parameters for KV cache allocation"
            )

        # Calculate block size based on available memory
        block_size = self.runner.block_size
        num_blocks = cache_bytes // (
            2  # Key and value
            * num_layers
            * block_size
            * num_heads
            * head_dim
            * torch.finfo(self.runner.dtype).bits
            // 8  # Size in bytes
        )

        if num_blocks == 0:
            raise RuntimeError("Not enough GPU memory to allocate KV cache")

        # Allocate KV cache
        shape = (2, num_layers, num_blocks, block_size, num_heads, head_dim)
        self.kv_cache = torch.zeros(
            shape,
            dtype=self.runner.dtype,
            device=self.runner.device,
        )

    def setup_shared_memory(self, name: Optional[str] = None) -> None:
        """Set up shared memory for inter-process communication.

        Args:
            name: Optional name for the shared memory block.
        """
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()

        # Create shared memory for communication
        if name is None:
            name = f"medvllm_shm_{os.getpid()}"

        # Calculate size needed for communication
        # This is a simplified example - adjust based on actual needs
        shm_size = 1024 * 1024  # 1MB default

        try:
            self.shm = SharedMemoryT(name=name, create=True, size=shm_size)
        except FileExistsError:
            # If shared memory with this name exists, try to open it
            self.shm = SharedMemoryT(name=name, create=False)

    def read_shm(self) -> Any:
        """Read data from shared memory.

        Returns:
            The deserialized data from shared memory.

        Raises:
            RuntimeError: If shared memory is not initialized
            json.JSONDecodeError: If the data is not valid JSON
            ValueError: If the data cannot be deserialized
        """
        if self.shm is None:
            raise RuntimeError("Shared memory not initialized")

        try:
            # Read the size of the data
            size_bytes = bytes(self.shm.buf[:8])
            size = int.from_bytes(size_bytes, byteorder="little")

            # Read the data
            data_bytes = bytes(self.shm.buf[8 : 8 + size])
            return json.loads(data_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError("Failed to deserialize data from shared memory") from e

    def write_shm(
        self, data: Union[BaseModel, Dict, List, str, int, float, bool, None]
    ) -> None:
        """Write data to shared memory.

        Args:
            data: The data to write to shared memory. Must be JSON-serializable.

        Raises:
            RuntimeError: If shared memory is not initialized
            TypeError: If the data is not JSON-serializable
            ValueError: If the data is too large for shared memory
        """
        if self.shm is None:
            raise RuntimeError("Shared memory not initialized")

        try:
            # Convert Pydantic models to dict if needed
            if hasattr(data, "model_dump"):
                data = data.model_dump()

            # Serialize data to JSON
            data_str = json.dumps(data)
            data_bytes = data_str.encode("utf-8")

            # Check if data fits in shared memory (leaving 8 bytes for size)
            max_size = len(self.shm.buf) - 8
            if len(data_bytes) > max_size:
                raise ValueError(
                    f"Data too large for shared memory: {len(data_bytes)} > {max_size}"
                )

            # Write size (8 bytes)
            size_bytes = len(data_bytes).to_bytes(8, byteorder="little")
            self.shm.buf[:8] = bytearray(size_bytes)

            # Write data
            self.shm.buf[8 : 8 + len(data_bytes)] = bytearray(data_bytes)

        except (TypeError, OverflowError) as e:
            raise TypeError(f"Data is not JSON-serializable: {e}") from e

        # Data is already written in the try block above
        pass

    def cleanup(self) -> None:
        """Clean up shared memory and KV cache."""
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None

        if self.kv_cache is not None:
            del self.kv_cache
            self.kv_cache = None

    def __del__(self) -> None:
        """Ensure resources are cleaned up when the object is destroyed."""
        self.cleanup()
