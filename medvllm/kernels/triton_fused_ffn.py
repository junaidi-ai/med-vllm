"""
Guarded placeholder Triton fused-FFN implementation.
- If Triton is available, provides a minimal fused GeLU MLP block.
- Otherwise, falls back to standard PyTorch eager implementation.

This is a scaffold for future optimization; correctness is prioritized over speed.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

_TRITON_AVAILABLE = False
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    # Allow forcing off via env for testing
    if os.getenv("MEDVLLM_DISABLE_TRITON", "0") == "1":
        return False
    return _TRITON_AVAILABLE


class EagerFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# Placeholder kernel: we still use nn.Linear but fuse activation pointwise to simulate a fused path
class TritonFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In a full Triton impl, matmuls would be fused; here we mimic with a custom GeLU kernel when possible
        h = self.fc1(x)
        h = _gelu_triton(h) if triton_available() else torch.nn.functional.gelu(h)
        y = self.fc2(h)
        return y


def _gelu_triton(x: torch.Tensor) -> torch.Tensor:
    # Minimal pointwise Triton kernel fallback to torch if tensor not CUDA or triton missing
    if not triton_available() or (not x.is_cuda):
        return torch.nn.functional.gelu(x)

    y = torch.empty_like(x)

    @triton.jit
    def gelu_kernel(X_PTR, Y_PTR, N_ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N_ELEMENTS
        x = tl.load(X_PTR + offs, mask=mask, other=0.0)
        # Approximate GELU
        c = 0.044715
        v = x * (1.0 + tl.math.tanh(0.7978845608 * (x + c * x * x * x))) * 0.5
        tl.store(Y_PTR + offs, v, mask=mask)

    n_elems = x.numel()
    BLOCK = 1024
    grid = lambda META: ((n_elems + META["BLOCK"] - 1) // META["BLOCK"],)
    gelu_kernel[grid](x, y, n_elems, BLOCK=BLOCK)
    return y


def build_fused_ffn_if_available(
    hidden_size: int, intermediate_size: int, bias: bool = True
) -> Optional[nn.Module]:
    if triton_available():
        return TritonFFN(hidden_size, intermediate_size, bias=bias)
    return None
