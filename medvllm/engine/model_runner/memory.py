from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel

from .types import *

if TYPE_CHECKING:
    from .base import ModelRunner


class MemoryManager:
    """Manages shared memory and KV cache for the model with medical optimizations.

    Features:
    - Medical domain-specific cache eviction policies
    - Distributed cache coherence
    - Cache statistics and monitoring
    - Memory-efficient storage for clinical text patterns
    """

    def __init__(self, runner: "ModelRunner") -> None:
        """Initialize the memory manager with medical optimizations.

        Args:
            runner: The parent ModelRunner instance.
        """
        self.runner = runner
        self.shm: Optional[SharedMemoryT] = None
        self.kv_cache: Optional[Dict[str, Any]] = None
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_used_blocks": 0,
            "total_blocks": 0,
        }
        self.medical_entities_cache: Dict[str, Any] = {}
        self.distributed_enabled = False

        # --- Tensor pooling (optional) ---
        self.pool_enabled: bool = bool(getattr(self.runner.config, "enable_memory_pooling", False))
        # Pool keyed by (device:str, dtype:str, bucket:int). Values are LIFO lists of tensors.
        self._pool: Dict[tuple[str, str, int], List[torch.Tensor]] = {}
        self._pool_bytes: int = 0
        self._pool_max_bytes: Optional[int] = getattr(self.runner.config, "pool_max_bytes", None)
        # auto device -> follow runner.device
        pool_dev_pref = getattr(self.runner.config, "pool_device", "auto")
        self._pool_device_str: str = (
            str(self.runner.device) if pool_dev_pref in (None, "auto") else str(pool_dev_pref)
        )
        # Pool stats
        self.pool_stats: Dict[str, int] = {
            "acquire_requests": 0,
            "acquire_hits": 0,
            "acquire_misses": 0,
            "released": 0,
            "reused": 0,
            "evicted": 0,
            "bytes": 0,
        }

    # -------- Tensor pooling helpers --------
    @staticmethod
    def _dtype_nbytes(dtype: torch.dtype) -> int:
        return (
            torch.finfo(dtype).bits // 8
            if dtype.is_floating_point
            else torch.iinfo(dtype).bits // 8
        )

    @staticmethod
    def _bucket_for(numel: int) -> int:
        # round up to next power-of-two bucket to improve reuse
        if numel <= 1:
            return 1
        b = 1
        while b < numel:
            b <<= 1
        return b

    def _key(self, device: torch.device, dtype: torch.dtype, bucket: int) -> tuple[str, str, int]:
        return (str(device), str(dtype), bucket)

    def _maybe_evict_until_under_limit(self) -> None:
        if self._pool_max_bytes is None:
            return
        while self._pool_bytes > self._pool_max_bytes and self._pool:
            # Evict from the largest bucket/device arbitrarily (simple heuristic)
            key = next(iter(self._pool))
            lst = self._pool[key]
            if not lst:
                del self._pool[key]
                continue
            t = lst.pop()
            try:
                self._pool_bytes -= t.numel() * self._dtype_nbytes(t.dtype)
            except Exception:
                pass
            self.pool_stats["evicted"] += 1
            if not lst:
                del self._pool[key]

    def acquire(
        self,
        shape: List[int] | tuple[int, ...],
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        init: str = "empty",  # "empty" | "zeros"
    ) -> torch.Tensor:
        """Acquire a tensor from the pool or allocate a new one.

        Pooling is enabled only when Config.enable_memory_pooling=True.
        """
        self.pool_stats["acquire_requests"] += 1
        device = device or self.runner.device
        dev_str = str(device)
        if self.pool_enabled and dev_str == self._pool_device_str:
            numel = 1
            for d in shape:
                numel *= int(d)
            bucket = self._bucket_for(numel)
            key = self._key(device, dtype, bucket)
            lst = self._pool.get(key, [])
            if lst:
                t = lst.pop()
                self.pool_stats["acquire_hits"] += 1
                self.pool_stats["reused"] += 1
                return t.view(*shape)
            else:
                self.pool_stats["acquire_misses"] += 1
        # Fallback: allocate new
        if init == "zeros":
            t = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        else:
            t = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        return t

    def release(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool for potential reuse."""
        if not isinstance(tensor, torch.Tensor):
            return
        self.pool_stats["released"] += 1
        if not self.pool_enabled:
            return
        # Only pool contiguous tensors on the configured pool device
        if str(tensor.device) != self._pool_device_str:
            return
        try:
            t = tensor.detach()
            if not t.is_contiguous():
                t = t.contiguous()
            numel = t.numel()
            bucket = self._bucket_for(numel)
            key = self._key(t.device, t.dtype, bucket)
            lst = self._pool.setdefault(key, [])
            lst.append(t)
            self._pool_bytes += numel * self._dtype_nbytes(t.dtype)
            self.pool_stats["bytes"] = self._pool_bytes
            self._maybe_evict_until_under_limit()
        except Exception:
            # Best-effort pooling
            pass

    def allocate_kv_cache(self, gpu_memory_utilization: float = 0.9) -> None:
        """Allocate GPU memory for the key-value cache with medical optimizations.

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

        # Initialize KV cache structure
        self.kv_cache = {
            "blocks": None,
            "block_size": 256,  # Default block size, can be tuned
            "free_blocks": [],
            "allocated_blocks": {},
            "block_metadata": {},
            "total_blocks": 0,
        }

        # Calculate number of blocks
        block_size_bytes = self.kv_cache["block_size"] * 2  # Assuming float16
        num_blocks = cache_bytes // block_size_bytes

        if num_blocks > 0:
            # Allocate blocks
            self.kv_cache["blocks"] = torch.empty(
                num_blocks * self.kv_cache["block_size"],
                dtype=torch.float16,
                device=self.runner.device,
            )
            self.kv_cache["free_blocks"] = list(range(num_blocks))
            self.kv_cache["total_blocks"] = num_blocks
            self.cache_stats["total_blocks"] = num_blocks

        # Get model config
        config = self.runner.model_config

        # Calculate shape for KV cache
        num_layers = getattr(config, "num_hidden_layers", 0)
        num_heads = getattr(config, "num_attention_heads", 0)
        head_dim = getattr(config, "head_dim", 0) or getattr(config, "hidden_size", 0) // num_heads

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
        kv_cache = torch.zeros(
            shape,
            dtype=self.runner.dtype,
            device=self.runner.device,
        )
        self.kv_cache = kv_cache  # type: ignore[assignment]

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

    def write_shm(self, data: Union[BaseModel, Dict, List, str, int, float, bool, None]) -> None:
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
            if hasattr(data, "model_dump") and data is not None:
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
        """Clean up shared memory, KV cache, and medical entity cache."""
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None

        if self.kv_cache is not None:
            if "blocks" in self.kv_cache and isinstance(self.kv_cache["blocks"], torch.Tensor):
                del self.kv_cache["blocks"]
            self.kv_cache = None

        # Clean up medical entities cache
        self.medical_entities_cache.clear()

        # Clear pool
        self._pool.clear()
        self._pool_bytes = 0
        self.pool_stats.update(
            {
                "acquire_requests": 0,
                "acquire_hits": 0,
                "acquire_misses": 0,
                "released": 0,
                "reused": 0,
                "evicted": 0,
                "bytes": 0,
            }
        )

        # Reset statistics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_used_blocks": 0,
            "total_blocks": self.cache_stats.get("total_blocks", 0),
        }

    def __del__(self) -> None:
        """Ensure resources are cleaned up when the object is destroyed."""
        self.cleanup()
