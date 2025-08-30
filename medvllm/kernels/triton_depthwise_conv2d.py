"""Guarded Triton depthwise conv2d with PyTorch fallback.

API:
- build_fused_depthwise_conv2d_if_available(C, K, stride=1, padding=1, dilation=1): returns nn.Module or None
- EagerDepthwiseConv2d(C, K, stride=1, padding=1, dilation=1): PyTorch depthwise conv module (groups=C)

The fused module exposes forward(x: [N,C,H,W]) -> [N,C,H,W].

Notes:
- Triton kernel targets CUDA only. If Triton/CUDA are unavailable, returns None so callers can fall back.
- Kernel computes per-channel KxK depthwise convolution. We use a simple tiling over output spatial domain and channels.
- Channels-last (NHWC memory format) compatibility: inputs in channels_last are converted to contiguous NCHW for the kernel and outputs are restored to channels_last for convenience. A true NHWC-optimized kernel is planned.
- Kernel supports arbitrary K (odd or even), stride>=1, dilation>=1, and padding provided by the caller.
"""

from __future__ import annotations

from typing import Optional, Tuple

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class EagerDepthwiseConv2d(nn.Module):
    def __init__(
        self,
        C: int,
        K: int,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 1,
        dilation: int | Tuple[int, int] = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.op = nn.Conv2d(
            C,
            C,
            kernel_size=K,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=C,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def _triton_available() -> bool:
    if os.getenv("MEDVLLM_DISABLE_TRITON", "0") == "1":
        return False
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


if _triton_available():
    import triton
    import triton.language as tl

    # Simple autotune across a few tile sizes
    _DWCONV_CONFIGS = [
        # Balanced tiles
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_warps=8, num_stages=2),
        # Rectangular tiles (H-dominant / W-dominant)
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_warps=8, num_stages=2),
        # Smaller tiles for small feature maps
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_warps=2, num_stages=2),
    ]

    @triton.autotune(configs=_DWCONV_CONFIGS, key=["H_out", "W_out"])  # type: ignore[arg-type]
    @triton.jit
    def _dwconv2d_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        N,
        C,
        H,
        W,
        K,
        stride,
        pad,
        dil,
        H_out,
        W_out,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_w = tl.program_id(1)
        pid_nc = tl.program_id(2)

        oh_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        ow_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

        nc = pid_nc  # 0..(N*C-1)
        n = nc // C
        c = nc % C

        # Compute mask for valid output positions
        oh_mask = oh_offsets < H_out
        ow_mask = ow_offsets < W_out
        ohw_valid = oh_mask[:, None] & ow_mask[None, :]

        # Accumulator
        acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

        # Precompute input base coordinates for this tile
        ih_base = oh_offsets * stride - pad
        iw_base = ow_offsets * stride - pad

        # Loop over KxK
        for kh in range(0, K):
            ih = ih_base + kh * dil
            ih_mask = (ih >= 0) & (ih < H)
            for kw in range(0, K):
                iw = iw_base + kw * dil
                iw_mask = (iw >= 0) & (iw < W)

                valid = ohw_valid & (ih_mask[:, None] & iw_mask[None, :])

                # Flattened input index: (((n*C + c)*H) + ih)*W + iw
                x_idx = ((n * C + c) * H + ih[:, None]) * W + iw[None, :]
                x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)

                # Weight index: (c*K + kh)*K + kw
                w_idx = (c * K + kh) * K + kw
                w_val = tl.load(w_ptr + w_idx)

                acc += x_val * w_val

        # Write output
        y_idx = ((n * C + c) * H_out + oh_offsets[:, None]) * W_out + ow_offsets[None, :]
        tl.store(y_ptr + y_idx, acc, mask=ohw_valid)

    # Specialized K=3 kernel with unrolled loops for reduced overhead
    @triton.autotune(configs=_DWCONV_CONFIGS, key=["H_out", "W_out"])  # type: ignore[arg-type]
    @triton.jit
    def _dwconv2d_kernel_k3(
        x_ptr,
        w_ptr,
        y_ptr,
        N,
        C,
        H,
        W,
        stride,
        pad,
        dil,
        H_out,
        W_out,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_w = tl.program_id(1)
        pid_nc = tl.program_id(2)

        oh_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        ow_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

        nc = pid_nc
        n = nc // C
        c = nc % C

        oh_mask = oh_offsets < H_out
        ow_mask = ow_offsets < W_out
        ohw_valid = oh_mask[:, None] & ow_mask[None, :]

        acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

        ih_base = oh_offsets * stride - pad
        iw_base = ow_offsets * stride - pad

        # kh = 0..2, kw = 0..2 (use static_range for compile-time unrolling)
        for kh in tl.static_range(0, 3):
            ih = ih_base + kh * dil
            ih_mask = (ih >= 0) & (ih < H)
            for kw in tl.static_range(0, 3):
                iw = iw_base + kw * dil
                iw_mask = (iw >= 0) & (iw < W)
                valid = ohw_valid & (ih_mask[:, None] & iw_mask[None, :])

                x_idx = ((n * C + c) * H + ih[:, None]) * W + iw[None, :]
                x_val = tl.where(valid, tl.load(x_ptr + x_idx, mask=valid, other=0.0), 0.0)

                w_idx = (c * 3 + kh) * 3 + kw
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

        y_idx = ((n * C + c) * H_out + oh_offsets[:, None]) * W_out + ow_offsets[None, :]
        tl.store(y_ptr + y_idx, acc, mask=ohw_valid)


class _TritonDepthwiseConv2d(nn.Module):
    def __init__(self, C: int, K: int, stride=1, padding=1, dilation=1):
        super().__init__()
        self.C = C
        self.K = K
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        # Weights per-channel [C, K, K]
        w = torch.randn(C, K, K, device='cuda') * (1.0 / (K * K))
        self.weight = nn.Parameter(w)
        # Eager fallback module for tiny shapes where Triton underperforms
        self._eager_fallback = EagerDepthwiseConv2d(
            C, K, stride=stride, padding=padding, dilation=dilation, bias=False
        ).to('cuda')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Triton fused depthwise conv requires CUDA tensor"
        x_in = x  # preserve original for possible eager fallback with memory format
        # Support channels-last inputs by converting to contiguous NCHW before kernel
        input_was_channels_last = False
        try:
            input_was_channels_last = x.is_contiguous(memory_format=torch.channels_last)
        except Exception:
            input_was_channels_last = False
        x = x.contiguous()  # ensure NCHW contiguous for Triton indexing
        N, C, H, W = x.shape
        assert C == self.C, "Input channels must match"
        K = self.K
        stride = self.stride
        pad = self.padding
        dil = self.dilation
        H_out = (H + 2 * pad - dil * (K - 1) - 1) // stride + 1
        W_out = (W + 2 * pad - dil * (K - 1) - 1) // stride + 1

        # Tiny-shape eager fallback (override with MEDVLLM_DWCONV_FORCE_TRITON=1)
        force_triton = os.getenv("MEDVLLM_DWCONV_FORCE_TRITON", "0") == "1"
        try:
            tiny_thresh = int(os.getenv("MEDVLLM_DWCONV_TINY_THRESHOLD", "4096"))
        except Exception:
            tiny_thresh = 4096
        if not force_triton and (H_out * W_out) < tiny_thresh:
            with torch.no_grad():
                # Sync weights to eager op: [C,K,K] -> [C,1,K,K]
                w_eager = self.weight.detach().reshape(C, 1, K, K).contiguous()
                self._eager_fallback.op.weight.copy_(w_eager)
            eager_in = x_in
            if input_was_channels_last:
                eager_in = eager_in.to(memory_format=torch.channels_last)
                self._eager_fallback = self._eager_fallback.to(memory_format=torch.channels_last)
            y_eager = self._eager_fallback(eager_in)
            return y_eager

        y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        # Flatten views for Triton
        x_flat = x
        w_flat = self.weight.reshape(-1)
        y_flat = y

        # Grid defined using autotuned meta-parameters BLOCK_H/BLOCK_W
        grid = lambda META: (
            triton.cdiv(H_out, META["BLOCK_H"]),
            triton.cdiv(W_out, META["BLOCK_W"]),
            N * C,
        )
        if K == 3:
            _dwconv2d_kernel_k3[grid](
                x_flat,
                w_flat,
                y_flat,
                N,
                C,
                H,
                W,
                stride,
                pad,
                dil,
                H_out,
                W_out,
            )
        else:
            _dwconv2d_kernel[grid](
                x_flat,
                w_flat,
                y_flat,
                N,
                C,
                H,
                W,
                K,
                stride,
                pad,
                dil,
                H_out,
                W_out,
            )
        # Restore channels_last memory format if input was channels_last for UX
        if input_was_channels_last:
            try:
                y = y.to(memory_format=torch.channels_last)
            except Exception:
                pass
        return y


def build_fused_depthwise_conv2d_if_available(
    C: int, K: int, *, stride=1, padding=1, dilation=1
) -> Optional[nn.Module]:
    """Return a fused depthwise conv2d module if Triton/CUDA is available, else None.

    Falls back to a thin wrapper around the eager op if Triton is present but no kernel is implemented yet.
    """
    if not isinstance(C, int) or not isinstance(K, int) or C <= 0 or K <= 0:
        return None
    if _triton_available():
        try:
            return _TritonDepthwiseConv2d(C, K, stride=stride, padding=padding, dilation=dilation)
        except Exception:
            return None
    return None
