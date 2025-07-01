from __future__ import annotations

import contextlib
import os
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.distributed as dist
from torch import Tensor
from typing_extensions import Literal, TypeAlias, TypedDict

if TYPE_CHECKING:
    from .types import _ModelRunnerT as ModelRunnerT

T = TypeVar("T")

# Type aliases for better type hints
ReductionOp = Literal["sum", "mean", "min", "max"]
DeviceType = Union[str, torch.device]


def validate_tensor(x: Any, name: str) -> None:
    """Validate that the input is a PyTorch tensor.

    Args:
        x: The input to validate.
        name: Name of the input for error messages.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(x)}")


def validate_positive_int(value: int, name: str) -> None:
    """Validate that the input is a positive integer.

    Args:
        value: The value to validate.
        name: Name of the parameter for error messages.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def validate_device(device: Union[str, torch.device]) -> torch.device:
    """Validate and convert device string to torch.device.

    Args:
        device: Device string or torch.device.

    Returns:
        Validated torch.device.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if not isinstance(device, torch.device):
        raise TypeError(f"device must be a string or torch.device, got {type(device)}")

    return device


def get_available_memory(device: Union[str, torch.device]) -> int:
    """Get available GPU memory in bytes.

    Args:
        device: The device to check memory for.

    Returns:
        Available memory in bytes.
    """
    device = validate_device(device)

    if device.type == "cuda":
        return torch.cuda.get_device_properties(
            device
        ).total_memory - torch.cuda.memory_allocated(device)

    # For CPU, return system available memory
    if hasattr(os, "sysconf"):
        if sys.platform == "darwin":
            # macOS
            import resource

            return cast(int, resource.getrlimit(resource.RLIMIT_DATA)[0])
        else:
            # Linux
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                    return (
                        int(meminfo.split("MemAvailable:")[1].split("kB")[0].strip())
                        * 1024
                    )
            except (FileNotFoundError, IndexError, ValueError):
                pass

    # Default fallback (4GB)
    return 4 * 1024**3


@contextlib.contextmanager
def set_default_tensor_type(dtype: torch.dtype) -> Generator[None, None, None]:
    """Context manager to temporarily set the default tensor type.

    Args:
        dtype: The dtype to set as default.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def synchronize() -> None:
    """Synchronize all processes."""
    if is_distributed():
        torch.distributed.barrier()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def all_reduce(tensor: Tensor, op: ReductionOp = "sum") -> Tensor:
    """All-reduce operation across all processes.

    Args:
        tensor: The tensor to reduce.
        op: The reduction operation ('sum', 'mean', 'min', 'max').

    Returns:
        The reduced tensor.
    """
    if not is_distributed():
        return tensor

    op = op.lower()
    if op == "sum":
        op = torch.distributed.ReduceOp.SUM
    elif op == "mean":
        op = torch.distributed.ReduceOp.SUM
        tensor = tensor.clone() / get_world_size()
    elif op == "min":
        op = torch.distributed.ReduceOp.MIN
    elif op == "max":
        op = torch.distributed.ReduceOp.MAX
    else:
        raise ValueError(f"Unsupported reduction operation: {op}")

    torch.distributed.all_reduce(tensor, op=op)
    return tensor
