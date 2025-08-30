from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False


_BIAS_GELU_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_H': 64}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_H': 128}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_H': 128}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_H': 256}, num_warps=2, num_stages=3),
    triton.Config({'BLOCK_H': 256}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_BIAS_GELU_AUTOTUNE_CONFIGS, key=['H'])
@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, y_ptr, N, H, stride_b, stride_h, BLOCK_H: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    x_row = x_ptr + pid_b * stride_b + h_offsets * stride_h
    y_row = y_ptr + pid_b * stride_b + h_offsets * stride_h

    x = tl.load(x_row, mask=mask_h, other=0.0)
    x_dtype = x.dtype
    x = x.to(tl.float32)
    if b_ptr is not None:
        b = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0).to(tl.float32)
        x = x + b

    # fast GELU approximation (tanh-based)
    k0 = 0.7978845608028654  # sqrt(2/pi)
    k1 = 0.044715
    x3 = x * x * x
    inner = k0 * (x + k1 * x3)
    # tanh(inner) via exp: tanh(z) = (e^{2z} - 1) / (e^{2z} + 1)
    e2 = tl.exp(2.0 * inner)
    tanh_inner = (e2 - 1.0) / (e2 + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner)
    y = y.to(x_dtype)
    tl.store(y_row, y, mask=mask_h)


class FusedBiasGELU(nn.Module):
    def __init__(self, hidden_size: int, bias: bool = True, block_h: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_bias = bias
        self.block_h = block_h
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "FusedBiasGELU expects CUDA tensor"
        BxN, H = x.shape[-2], x.shape[-1]
        y = torch.empty_like(x)
        B = BxN  # treat leading dims flattened per row

        def grid(meta):
            return (B, (H + meta["BLOCK_H"] - 1) // meta["BLOCK_H"])

        b_ptr = None
        if self.use_bias and self.bias is not None:
            b_ptr = (
                self.bias if self.bias.device == x.device else self.bias.to(x.device)
            ).contiguous()
        _bias_gelu_kernel[grid](
            x,
            b_ptr,
            y,
            B,
            H,
            x.stride(-2),
            x.stride(-1),
            # BLOCK_H, num_warps, num_stages picked by autotuner
        )
        return y


def build_fused_bias_gelu_if_available(
    hidden_size: int, *, bias: bool = True
) -> Optional[nn.Module]:
    """Return a fused Bias+GELU module if Triton+CUDA and env flag enabled; else None."""
    if not _HAVE_TRITON:
        return None
    if not torch.cuda.is_available():
        return None
    if os.getenv("MEDVLLM_ENABLE_TRITON_BIAS_GELU", "0") != "1":
        return None
    return FusedBiasGELU(hidden_size, bias=bias)
