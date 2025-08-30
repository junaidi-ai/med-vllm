from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False


# Autotune presets (persisted winners from benchmarks/sweep_sep3d_autotune.py)
# Source report: reports/sep3d_autotune_2025-08-30.json
_SEP3D_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_W': 128}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_W': 256}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_W': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=4),
]

# Optional: allow forcing a single config via env to profile/prune easily
try:
    _bw = int(os.getenv("MEDVLLM_SEP3D_BLOCK_W", "0"))
    _nw = int(os.getenv("MEDVLLM_SEP3D_WARPS", "0"))
    _ns = int(os.getenv("MEDVLLM_SEP3D_STAGES", "0"))
    if _bw in (64, 128, 256) and _nw in (2, 4, 8) and _ns in (2, 3, 4):
        _SEP3D_AUTOTUNE_CONFIGS = [triton.Config({'BLOCK_W': _bw}, num_warps=_nw, num_stages=_ns)]
except Exception:
    pass


@triton.autotune(configs=_SEP3D_AUTOTUNE_CONFIGS, key=['D', 'H', 'W'])
@triton.jit
def _dwconv3d3x3x3_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    bias_ptr,
    B,
    C,
    D,
    H,
    W,
    sx_b,
    sx_c,
    sx_d,
    sx_h,
    sx_w,
    sw_c,
    sw_kd,
    sw_kh,
    sw_kw,
    sy_b,
    sy_c,
    sy_d,
    sy_h,
    sy_w,
    FUSE_BIAS: tl.constexpr,
    FUSE_RELU: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_wt = tl.program_id(2)

    b = pid_bc // C
    c = pid_bc % C
    d = pid_dh // H
    h = pid_dh % H
    w0 = pid_wt * BLOCK_W
    w_offs = w0 + tl.arange(0, BLOCK_W)

    in_d0 = d - 1
    in_h0 = h - 1
    in_w0 = w_offs - 1

    mask_w = w_offs < W

    # Base pointers
    x_base = x_ptr + b * sx_b + c * sx_c
    y_row = y_ptr + b * sy_b + c * sy_c + d * sy_d + h * sy_h + w_offs * sy_w

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # (removed) alignment hints; not applicable to constexpr/scalars here

    # Prefetch the 3x3x3 weight for this channel c into registers (fp32)
    w00 = tl.load(w_ptr + c * sw_c + 0 * sw_kd + 0 * sw_kh + 0 * sw_kw).to(tl.float32)
    w01 = tl.load(w_ptr + c * sw_c + 0 * sw_kd + 0 * sw_kh + 1 * sw_kw).to(tl.float32)
    w02 = tl.load(w_ptr + c * sw_c + 0 * sw_kd + 0 * sw_kh + 2 * sw_kw).to(tl.float32)
    w10 = tl.load(w_ptr + c * sw_c + 0 * sw_kd + 1 * sw_kh + 0 * sw_kw).to(tl.float32)
    w11 = tl.load(w_ptr + c * sw_c + 0 * sw_kd + 1 * sw_kh + 1 * sw_kw).to(tl.float32)
    w12 = tl.load(w_ptr + c * sw_c + 0 * sw_kd + 1 * sw_kh + 2 * sw_kw).to(tl.float32)
    w20 = tl.load(w_ptr + c * sw_c + 0 * sw_kd + 2 * sw_kh + 0 * sw_kw).to(tl.float32)
    w21 = tl.load(w_ptr + c * sw_c + 0 * sw_kd + 2 * sw_kh + 1 * sw_kw).to(tl.float32)
    w22 = tl.load(w_ptr + c * sw_c + 0 * sw_kd + 2 * sw_kh + 2 * sw_kw).to(tl.float32)

    w30 = tl.load(w_ptr + c * sw_c + 1 * sw_kd + 0 * sw_kh + 0 * sw_kw).to(tl.float32)
    w31 = tl.load(w_ptr + c * sw_c + 1 * sw_kd + 0 * sw_kh + 1 * sw_kw).to(tl.float32)
    w32 = tl.load(w_ptr + c * sw_c + 1 * sw_kd + 0 * sw_kh + 2 * sw_kw).to(tl.float32)
    w40 = tl.load(w_ptr + c * sw_c + 1 * sw_kd + 1 * sw_kh + 0 * sw_kw).to(tl.float32)
    w41 = tl.load(w_ptr + c * sw_c + 1 * sw_kd + 1 * sw_kh + 1 * sw_kw).to(tl.float32)
    w42 = tl.load(w_ptr + c * sw_c + 1 * sw_kd + 1 * sw_kh + 2 * sw_kw).to(tl.float32)
    w50 = tl.load(w_ptr + c * sw_c + 1 * sw_kd + 2 * sw_kh + 0 * sw_kw).to(tl.float32)
    w51 = tl.load(w_ptr + c * sw_c + 1 * sw_kd + 2 * sw_kh + 1 * sw_kw).to(tl.float32)
    w52 = tl.load(w_ptr + c * sw_c + 1 * sw_kd + 2 * sw_kh + 2 * sw_kw).to(tl.float32)

    w60 = tl.load(w_ptr + c * sw_c + 2 * sw_kd + 0 * sw_kh + 0 * sw_kw).to(tl.float32)
    w61 = tl.load(w_ptr + c * sw_c + 2 * sw_kd + 0 * sw_kh + 1 * sw_kw).to(tl.float32)
    w62 = tl.load(w_ptr + c * sw_c + 2 * sw_kd + 0 * sw_kh + 2 * sw_kw).to(tl.float32)
    w70 = tl.load(w_ptr + c * sw_c + 2 * sw_kd + 1 * sw_kh + 0 * sw_kw).to(tl.float32)
    w71 = tl.load(w_ptr + c * sw_c + 2 * sw_kd + 1 * sw_kh + 1 * sw_kw).to(tl.float32)
    w72 = tl.load(w_ptr + c * sw_c + 2 * sw_kd + 1 * sw_kh + 2 * sw_kw).to(tl.float32)
    w80 = tl.load(w_ptr + c * sw_c + 2 * sw_kd + 2 * sw_kh + 0 * sw_kw).to(tl.float32)
    w81 = tl.load(w_ptr + c * sw_c + 2 * sw_kd + 2 * sw_kh + 1 * sw_kw).to(tl.float32)
    w82 = tl.load(w_ptr + c * sw_c + 2 * sw_kd + 2 * sw_kh + 2 * sw_kw).to(tl.float32)

    # Fast path if this W-tile is strictly interior (no left/right halo touches)
    interior = (w0 > 0) & ((w0 + BLOCK_W) < (W - 1))

    for kd in tl.static_range(0, 3):
        id_ = in_d0 + kd
        mask_d = (0 <= id_) & (id_ < D)
        for kh in tl.static_range(0, 3):
            ih = in_h0 + kh
            mask_h_ = (0 <= ih) & (ih < H)
            base = x_base + id_ * sx_d + ih * sx_h
            idx = tl.arange(0, BLOCK_W)
            g0 = w0 - 1 + idx
            if interior:
                if sx_w == 1:
                    # Apply a stricter alignment guard: only hint wider contiguity when
                    # the starting column index is 8-element aligned. This approximates
                    # 16B alignment for fp16/bf16 and keeps scalar-safe behavior otherwise.
                    align_ok = (w0 & 7) == 0
                    if align_ok:
                        tl.max_contiguous(g0, 8)
                        x0 = tl.load(base + (g0 + 0) * sx_w).to(tl.float32)
                        x1 = tl.load(base + (g0 + 1) * sx_w).to(tl.float32)
                        x2 = tl.load(base + (g0 + 2) * sx_w).to(tl.float32)
                    else:
                        m01 = mask_w & mask_d & mask_h_
                        ml0 = m01 & (g0 + 0 >= 0) & (g0 + 0 < W)
                        ml1 = m01 & (g0 + 1 >= 0) & (g0 + 1 < W)
                        ml2 = m01 & (g0 + 2 >= 0) & (g0 + 2 < W)
                        x0 = tl.load(base + (g0 + 0) * sx_w, mask=ml0, other=0.0).to(tl.float32)
                        x1 = tl.load(base + (g0 + 1) * sx_w, mask=ml1, other=0.0).to(tl.float32)
                        x2 = tl.load(base + (g0 + 2) * sx_w, mask=ml2, other=0.0).to(tl.float32)
            else:
                m01 = mask_w & mask_d & mask_h_
                ml0 = m01 & (g0 + 0 >= 0) & (g0 + 0 < W)
                ml1 = m01 & (g0 + 1 >= 0) & (g0 + 1 < W)
                ml2 = m01 & (g0 + 2 >= 0) & (g0 + 2 < W)
                x0 = tl.load(base + (g0 + 0) * sx_w, mask=ml0, other=0.0).to(tl.float32)
                x1 = tl.load(base + (g0 + 1) * sx_w, mask=ml1, other=0.0).to(tl.float32)
                x2 = tl.load(base + (g0 + 2) * sx_w, mask=ml2, other=0.0).to(tl.float32)
            # Select prefetched weights
            if kd == 0 and kh == 0:
                acc += x0 * w00 + x1 * w01 + x2 * w02
            elif kd == 0 and kh == 1:
                acc += x0 * w10 + x1 * w11 + x2 * w12
            elif kd == 0 and kh == 2:
                acc += x0 * w20 + x1 * w21 + x2 * w22
            elif kd == 1 and kh == 0:
                acc += x0 * w30 + x1 * w31 + x2 * w32
            elif kd == 1 and kh == 1:
                acc += x0 * w40 + x1 * w41 + x2 * w42
            elif kd == 1 and kh == 2:
                acc += x0 * w50 + x1 * w51 + x2 * w52
            elif kd == 2 and kh == 0:
                acc += x0 * w60 + x1 * w61 + x2 * w62
            elif kd == 2 and kh == 1:
                acc += x0 * w70 + x1 * w71 + x2 * w72
            else:  # kd == 2 and kh == 2
                acc += x0 * w80 + x1 * w81 + x2 * w82

    out = acc
    if FUSE_BIAS:
        # add channel-wise bias
        bval = tl.load(bias_ptr + c)
        out = out + bval
    if FUSE_RELU:
        out = tl.maximum(out, 0.0)
    tl.store(y_row, out.to(tl.float32), mask=mask_w)


class DepthwiseConv3d3x3x3(nn.Module):
    """
    Depthwise 3D conv (3x3x3) with padding=1, stride=1, groups=C.
    Expects weight of shape [C, 3, 3, 3]. Bias is optional.
    """

    def __init__(self, channels: int, bias: bool = False, block_w: int = 128):
        super().__init__()
        self.channels = channels
        self.block_w = block_w
        self.weight = nn.Parameter(torch.randn(channels, 3, 3, 3) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        if not (x.is_cuda and _HAVE_TRITON):
            # Fallback: use PyTorch groups=C conv3d
            return self._fallback_eager(x)
        B, C, D, H, W = x.shape
        assert C == self.channels
        y = torch.empty_like(x, dtype=torch.float32)

        def grid(meta):
            return (B * C, D * H, (W + meta["BLOCK_W"] - 1) // meta["BLOCK_W"])

        # Ensure kernel sees CUDA pointers on the right device
        w_ptr = self.weight if self.weight.device == x.device else self.weight.to(x.device)
        w_ptr = w_ptr.contiguous()

        fuse_bias = os.getenv("MEDVLLM_SEP3D_FUSE_BIAS", "0") == "1" and (self.bias is not None)
        fuse_relu = os.getenv("MEDVLLM_SEP3D_FUSE_RELU", "0") == "1"
        bias_ptr = None
        if fuse_bias:
            b = self.bias if self.bias.device == x.device else self.bias.to(x.device)
            bias_ptr = b.contiguous()

        _dwconv3d3x3x3_kernel[grid](
            x,
            w_ptr,
            y,
            bias_ptr if fuse_bias else x,  # dummy valid pointer if unused
            B,
            C,
            D,
            H,
            W,
            *x.stride(),
            *w_ptr.stride(),
            *y.stride(),
            fuse_bias,
            fuse_relu,
        )
        if (self.bias is not None) and (not fuse_bias):
            b = self.bias if self.bias.device == x.device else self.bias.to(x.device)
            y = y + b.view(1, C, 1, 1, 1)
        return y.to(x.dtype)

    def _fallback_eager(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        w = self.weight
        bias = self.bias
        # Convert to conv3d weight format [C, 1, 3,3,3] with groups=C
        w_conv = w.view(C, 1, 3, 3, 3)
        y = torch.nn.functional.conv3d(x, w_conv, bias=bias, stride=1, padding=1, groups=C)
        return y


def build_fused_separable_conv3d_if_available(
    channels: int, *, bias: bool = False
) -> Optional[nn.Module]:
    if not _HAVE_TRITON:
        return None
    if not torch.cuda.is_available():
        return None
    if os.getenv("MEDVLLM_ENABLE_TRITON_SEP3D", "0") != "1":
        return None
    return DepthwiseConv3d3x3x3(channels, bias=bias)
