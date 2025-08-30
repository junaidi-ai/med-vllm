"""
Guarded placeholder Triton fused softmax×V.
- If Triton is available on CUDA, uses a Triton row-wise softmax kernel, then matmul with V.
- Else falls back to torch.softmax + torch.matmul.

This is a scaffold for future full fusion (prob*V) kernel.

Default behavior: the Triton fused path is DISABLED unless explicitly enabled via
MEDVLLM_ENABLE_TRITON_SOFTMAXV=1. This is because the current placeholder may be
slower than eager/SDPA in common CUDA settings.
"""

from __future__ import annotations

import os
from typing import Optional
import torch
import triton
import triton.language as tl
from torch import nn

_TRITON_AVAILABLE = False
try:
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    if os.getenv("MEDVLLM_DISABLE_TRITON", "0") == "1":
        return False
    return _TRITON_AVAILABLE


@triton.jit
def _row_softmax_kernel(X, Y, ROWS: tl.constexpr, COLS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * COLS + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < COLS
    x = tl.load(X + offs, mask=mask, other=-float("inf"))
    # subtract max for stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    ex = tl.exp(x)
    den = tl.sum(ex, axis=0) + 1e-9
    y = ex / den
    tl.store(Y + offs, y, mask=mask)


## Autotune configuration lists (full vs fast-compile)
STREAMING_CONFIGS_FULL = [
    # BLOCK_N=64 variants (S up to 2048 -> tiles up to 32)
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_D": 64, "MAX_TILES": 8, "K": 4}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_D": 64, "MAX_TILES": 16, "K": 4}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_D": 128, "MAX_TILES": 16, "K": 4}, num_warps=8, num_stages=2
    ),
    # BLOCK_N=128 variants (S up to 2048 -> tiles up to 16)
    triton.Config(
        {"BLOCK_N": 128, "BLOCK_D": 32, "MAX_TILES": 4, "K": 4}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 128, "BLOCK_D": 64, "MAX_TILES": 4, "K": 4}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 128, "BLOCK_D": 64, "MAX_TILES": 8, "K": 4}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 128, "BLOCK_D": 128, "MAX_TILES": 8, "K": 4}, num_warps=8, num_stages=2
    ),
    # BLOCK_N=256 variants (S up to 2048 -> tiles up to 8)
    triton.Config(
        {"BLOCK_N": 256, "BLOCK_D": 64, "MAX_TILES": 4, "K": 4}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 256, "BLOCK_D": 128, "MAX_TILES": 4, "K": 4}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 256, "BLOCK_D": 64, "MAX_TILES": 8, "K": 4}, num_warps=8, num_stages=1
    ),
]

# Single minimal config for fast compile validation
STREAMING_CONFIGS_FAST = [
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_D": 64, "MAX_TILES": 4, "K": 4}, num_warps=4, num_stages=1
    ),
]

# Narrow preset to reduce JIT time while exploring a few choices
STREAMING_CONFIGS_NARROW = [
    # K=4
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_D": 32, "MAX_TILES": 8, "K": 4}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_D": 64, "MAX_TILES": 8, "K": 4}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_D": 128, "MAX_TILES": 8, "K": 4}, num_warps=8, num_stages=2
    ),
    # K=8
    triton.Config(
        {"BLOCK_N": 128, "BLOCK_D": 32, "MAX_TILES": 4, "K": 8}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 128, "BLOCK_D": 64, "MAX_TILES": 4, "K": 8}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_N": 128, "BLOCK_D": 128, "MAX_TILES": 4, "K": 8}, num_warps=8, num_stages=2
    ),
]

if os.getenv("MEDVLLM_SOFTMAXV_COMPILE_FAST"):
    STREAMING_CONFIGS = STREAMING_CONFIGS_FAST
elif os.getenv("MEDVLLM_SOFTMAXV_COMPILE_NARROW"):
    STREAMING_CONFIGS = STREAMING_CONFIGS_NARROW
else:
    STREAMING_CONFIGS = STREAMING_CONFIGS_FULL

# Optionally force a single config by index to avoid JIT looping
_force_idx_env = os.getenv("MEDVLLM_SOFTMAXV_FORCE_CONFIG")
if _force_idx_env is not None:
    try:
        _idx = int(_force_idx_env)
        if 0 <= _idx < len(STREAMING_CONFIGS):
            STREAMING_CONFIGS = [STREAMING_CONFIGS[_idx]]
    except Exception:
        pass


# Fused streaming kernel: computes softmax(scores[row,:]) @ V without materializing probs
@triton.autotune(
    configs=STREAMING_CONFIGS,
    key=["S", "D"],  # sequence len and head dim
)
@triton.jit
def _softmaxv_streaming(
    scores_ptr,
    v_ptr,
    y_ptr,
    B,
    H,
    S,
    D,
    s_sc_b,
    s_sc_h,
    s_sc_row,
    s_sc_col,
    s_v_b,
    s_v_h,
    s_v_s,
    s_v_d,
    s_y_b,
    s_y_h,
    s_y_s,
    s_y_d,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_TILES: tl.constexpr,
    K: tl.constexpr,
):
    # program ids
    pid_bh = tl.program_id(0)  # over B*H
    pid_row = tl.program_id(1)  # over S rows
    pid_d = tl.program_id(2)  # over D tiles

    b = pid_bh // H
    h = pid_bh % H
    row = pid_row
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Base pointers
    # scores layout [B,H,S,S]
    scores_row_ptr = scores_ptr + b * s_sc_b + h * s_sc_h + row * s_sc_row
    # V layout [B,H,S,D]
    v_base_ptr = v_ptr + b * s_v_b + h * s_v_h
    # Output [B,H,S,D]
    y_base_ptr = y_ptr + b * s_y_b + h * s_y_h

    # Single-pass online softmax with vectorized V accumulation over tiles [BLOCK_N, BLOCK_D]
    EPS = 1e-9
    m = tl.full((), -float("inf"), dtype=tl.float32)
    l = tl.zeros((), dtype=tl.float32)
    a = tl.zeros((BLOCK_D,), dtype=tl.float32)

    S_i = tl.full((), S, dtype=tl.int32)
    tl.multiple_of(d_offs, 16)
    tl.max_contiguous(d_offs, 16)
    for t in tl.static_range(0, MAX_TILES):
        n0 = t * BLOCK_N
        # process elements within the tile in chunks of K (compile-time unrolled)
        for k0 in tl.static_range(0, BLOCK_N, K):
            # unroll K scalar rows
            # NOTE: K should divide BLOCK_N in configs to keep bounds simple
            n = n0 + k0
            # Prefetch K scores and K V vectors into registers (on-chip)
            n_mask0 = n < S_i
            s_val0 = tl.load(scores_row_ptr + n * s_sc_col, mask=n_mask0, other=-float("inf"))
            v_ptr_vec0 = v_base_ptr + n * s_v_s + d_offs * s_v_d
            v_vec0 = tl.load(
                v_ptr_vec0, mask=(n_mask0 & d_mask), other=0.0, cache_modifier=".ca"
            ).to(tl.float32)

            n1 = n + 1
            n_mask1 = n1 < S_i
            s_val1 = tl.load(scores_row_ptr + n1 * s_sc_col, mask=n_mask1, other=-float("inf"))
            v_ptr_vec1 = v_base_ptr + n1 * s_v_s + d_offs * s_v_d
            v_vec1 = tl.load(
                v_ptr_vec1, mask=(n_mask1 & d_mask), other=0.0, cache_modifier=".ca"
            ).to(tl.float32)

            n2 = n1 + 1
            n_mask2 = n2 < S_i
            s_val2 = tl.load(scores_row_ptr + n2 * s_sc_col, mask=n_mask2, other=-float("inf"))
            v_ptr_vec2 = v_base_ptr + n2 * s_v_s + d_offs * s_v_d
            v_vec2 = tl.load(
                v_ptr_vec2, mask=(n_mask2 & d_mask), other=0.0, cache_modifier=".ca"
            ).to(tl.float32)

            n3 = n2 + 1
            n_mask3 = n3 < S_i
            s_val3 = tl.load(scores_row_ptr + n3 * s_sc_col, mask=n_mask3, other=-float("inf"))
            v_ptr_vec3 = v_base_ptr + n3 * s_v_s + d_offs * s_v_d
            v_vec3 = tl.load(
                v_ptr_vec3, mask=(n_mask3 & d_mask), other=0.0, cache_modifier=".ca"
            ).to(tl.float32)

            # Sequentially apply online softmax updates using staged data
            m_new = tl.maximum(m, s_val0)
            scale_old = tl.exp(m - m_new)
            e0 = tl.exp((s_val0 - m_new).to(tl.float32))
            a = a * scale_old + v_vec0 * e0
            l = l * scale_old + e0
            m = m_new

            m_new = tl.maximum(m, s_val1)
            scale_old = tl.exp(m - m_new)
            e1 = tl.exp((s_val1 - m_new).to(tl.float32))
            a = a * scale_old + v_vec1 * e1
            l = l * scale_old + e1
            m = m_new

            m_new = tl.maximum(m, s_val2)
            scale_old = tl.exp(m - m_new)
            e2 = tl.exp((s_val2 - m_new).to(tl.float32))
            a = a * scale_old + v_vec2 * e2
            l = l * scale_old + e2
            m = m_new

            m_new = tl.maximum(m, s_val3)
            scale_old = tl.exp(m - m_new)
            e3 = tl.exp((s_val3 - m_new).to(tl.float32))
            a = a * scale_old + v_vec3 * e3
            l = l * scale_old + e3
            m = m_new

    # write normalized output tile
    y_ptr_tile = y_base_ptr + row * s_y_s + d_offs * s_y_d
    out = a / (l + EPS)
    tl.store(y_ptr_tile, out, mask=d_mask)


@triton.jit
def _softmaxv_streaming_noauto(
    scores_ptr,
    v_ptr,
    y_ptr,
    B,
    H,
    S,
    D,
    s_sc_b,
    s_sc_h,
    s_sc_row,
    s_sc_col,
    s_v_b,
    s_v_h,
    s_v_s,
    s_v_d,
    s_y_b,
    s_y_h,
    s_y_s,
    s_y_d,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_TILES: tl.constexpr,
    K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H
    row = pid_row
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    scores_row_ptr = scores_ptr + b * s_sc_b + h * s_sc_h + row * s_sc_row
    v_base_ptr = v_ptr + b * s_v_b + h * s_v_h
    y_base_ptr = y_ptr + b * s_y_b + h * s_y_h

    EPS = 1e-9
    m = tl.full((), -float("inf"), dtype=tl.float32)
    l = tl.zeros((), dtype=tl.float32)
    a = tl.zeros((BLOCK_D,), dtype=tl.float32)

    S_i = tl.full((), S, dtype=tl.int32)
    tl.multiple_of(d_offs, 16)
    tl.max_contiguous(d_offs, 16)
    for t in tl.static_range(0, MAX_TILES):
        n0 = t * BLOCK_N
        for k0 in tl.static_range(0, BLOCK_N, K):
            n = n0 + k0
            # Prefetch up to 4 rows (assuming K>=4; extra rows are masked via S bounds)
            n_mask0 = n < S_i
            s_val0 = tl.load(scores_row_ptr + n * s_sc_col, mask=n_mask0, other=-float("inf"))
            v_vec0 = tl.load(
                v_base_ptr + n * s_v_s + d_offs * s_v_d,
                mask=(n_mask0 & d_mask),
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)

            n1 = n + 1
            n_mask1 = n1 < S_i
            s_val1 = tl.load(scores_row_ptr + n1 * s_sc_col, mask=n_mask1, other=-float("inf"))
            v_vec1 = tl.load(
                v_base_ptr + n1 * s_v_s + d_offs * s_v_d,
                mask=(n_mask1 & d_mask),
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)

            n2 = n1 + 1
            n_mask2 = n2 < S_i
            s_val2 = tl.load(scores_row_ptr + n2 * s_sc_col, mask=n_mask2, other=-float("inf"))
            v_vec2 = tl.load(
                v_base_ptr + n2 * s_v_s + d_offs * s_v_d,
                mask=(n_mask2 & d_mask),
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)

            n3 = n2 + 1
            n_mask3 = n3 < S_i
            s_val3 = tl.load(scores_row_ptr + n3 * s_sc_col, mask=n_mask3, other=-float("inf"))
            v_vec3 = tl.load(
                v_base_ptr + n3 * s_v_s + d_offs * s_v_d,
                mask=(n_mask3 & d_mask),
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)

            m_new = tl.maximum(m, s_val0)
            scale_old = tl.exp(m - m_new)
            e0 = tl.exp((s_val0 - m_new).to(tl.float32))
            a = a * scale_old + v_vec0 * e0
            l = l * scale_old + e0
            m = m_new

            m_new = tl.maximum(m, s_val1)
            scale_old = tl.exp(m - m_new)
            e1 = tl.exp((s_val1 - m_new).to(tl.float32))
            a = a * scale_old + v_vec1 * e1
            l = l * scale_old + e1
            m = m_new

            m_new = tl.maximum(m, s_val2)
            scale_old = tl.exp(m - m_new)
            e2 = tl.exp((s_val2 - m_new).to(tl.float32))
            a = a * scale_old + v_vec2 * e2
            l = l * scale_old + e2
            m = m_new

            m_new = tl.maximum(m, s_val3)
            scale_old = tl.exp(m - m_new)
            e3 = tl.exp((s_val3 - m_new).to(tl.float32))
            a = a * scale_old + v_vec3 * e3
            l = l * scale_old + e3
            m = m_new

    y_ptr_tile = y_base_ptr + row * s_y_s + d_offs * s_y_d
    out = a / (l + EPS)
    tl.store(y_ptr_tile, out, mask=d_mask)


class SoftmaxV(nn.Module):
    """
    Compute softmax(scores) @ V with scores [B,H,S,S] and V [B,H,S,D].
    """

    def forward(self, scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Eager baseline
        prob = torch.softmax(scores, dim=-1)
        return torch.matmul(prob, v)


class TritonSoftmaxV(nn.Module):
    def forward(self, scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if not (triton_available() and scores.is_cuda and v.is_cuda):
            prob = torch.softmax(scores, dim=-1)
            return torch.matmul(prob, v)
        # Use streaming kernel only when explicitly enabled (experimental)
        if os.getenv("MEDVLLM_ENABLE_TRITON_SOFTMAXV_STREAMING", "0") == "1":
            assert scores.dim() == 4 and v.dim() == 4, "scores [B,H,S,S], v [B,H,S,D]"
            B, H, S, S2 = scores.shape
            assert S == S2, "scores last dims must be SxS"
            D = v.shape[-1]
            y = torch.empty((B, H, S, D), device=scores.device, dtype=v.dtype)

            # Heuristic: skip streaming for small shapes unless forced
            min_s = int(os.getenv("MEDVLLM_SOFTMAXV_STREAMING_MIN_S", "2048"))
            min_d = int(os.getenv("MEDVLLM_SOFTMAXV_STREAMING_MIN_D", "128"))
            force_stream = os.getenv("MEDVLLM_FORCE_STREAMING_SOFTMAXV", "0") == "1"
            if not force_stream and (S < min_s or D < min_d):
                # Fallback to default Triton path (row-softmax + matmul)
                B_, H_, S_, _ = scores.shape
                prob = torch.empty_like(scores)
                scores_2d = scores.reshape(-1, S_, S_)
                prob_2d = prob.reshape(-1, S_, S_)
                for i in range(scores_2d.shape[0]):
                    x = scores_2d[i]
                    y2 = prob_2d[i]
                    BLOCK = ((S_ + 127) // 128) * 128
                    grid2 = (S_,)
                    _row_softmax_kernel[grid2](x, y2, ROWS=S_, COLS=S_, BLOCK=BLOCK)
                return torch.matmul(prob, v)

            # Strides (in elements) for indexing: full 4D strides
            sc_b, sc_h, sc_row, sc_col = scores.stride()
            v_b, v_h, v_s, v_d = v.stride()
            y_b, y_h, y_s, y_d = y.stride()

            # Grid: (B*H, S rows, D tiles)
            def grid(meta):
                return (
                    B * H,
                    S,
                    (D + meta["BLOCK_D"] - 1) // meta["BLOCK_D"],
                )

            # Optional no-autotune path to avoid autotune compile stalls
            if os.getenv("MEDVLLM_SOFTMAXV_NO_AUTOTUNE", "0") == "1":
                BLOCK_N = int(os.getenv("MEDVLLM_SOFTMAXV_BLOCK_N", "64"))
                BLOCK_D = int(os.getenv("MEDVLLM_SOFTMAXV_BLOCK_D", "64"))
                K = int(os.getenv("MEDVLLM_SOFTMAXV_K", "4"))
                NUM_WARPS = int(os.getenv("MEDVLLM_SOFTMAXV_NUM_WARPS", "4"))
                NUM_STAGES = int(os.getenv("MEDVLLM_SOFTMAXV_NUM_STAGES", "2"))
                MAX_TILES_CAP = int(os.getenv("MEDVLLM_SOFTMAXV_MAX_TILES_CAP", "32"))
                # compute MAX_TILES compile-time bound
                max_tiles = (S + BLOCK_N - 1) // BLOCK_N
                max_tiles = min(max_tiles, MAX_TILES_CAP)

                def grid_noauto(meta):
                    return (
                        B * H,
                        S,
                        (D + BLOCK_D - 1) // BLOCK_D,
                    )

                _softmaxv_streaming_noauto[grid_noauto](
                    scores,
                    v,
                    y,
                    B,
                    H,
                    S,
                    D,
                    sc_b,
                    sc_h,
                    sc_row,
                    sc_col,
                    v_b,
                    v_h,
                    v_s,
                    v_d,
                    y_b,
                    y_h,
                    y_s,
                    y_d,
                    BLOCK_N=BLOCK_N,
                    BLOCK_D=BLOCK_D,
                    MAX_TILES=max_tiles,
                    K=K,
                    num_warps=NUM_WARPS,
                    num_stages=NUM_STAGES,
                )
            else:
                _softmaxv_streaming[grid](
                    scores,
                    v,
                    y,
                    B,
                    H,
                    S,
                    D,
                    sc_b,
                    sc_h,
                    sc_row,
                    sc_col,
                    v_b,
                    v_h,
                    v_s,
                    v_d,
                    y_b,
                    y_h,
                    y_s,
                    y_d,
                )
            return y
        # Default Triton path: row-softmax kernel + matmul
        B, H, S, _ = scores.shape
        prob = torch.empty_like(scores)
        scores_2d = scores.reshape(-1, S, S)
        prob_2d = prob.reshape(-1, S, S)
        for i in range(scores_2d.shape[0]):
            x = scores_2d[i]
            y = prob_2d[i]
            BLOCK = ((S + 127) // 128) * 128
            grid = (S,)
            _row_softmax_kernel[grid](x, y, ROWS=S, COLS=S, BLOCK=BLOCK)
        return torch.matmul(prob, v)


def build_fused_softmaxv_if_available() -> Optional[nn.Module]:
    """Return TritonSoftmaxV only when explicitly enabled.

    Enable via environment variable: MEDVLLM_ENABLE_TRITON_SOFTMAXV=1
    Returns None otherwise.
    """
    if os.getenv("MEDVLLM_ENABLE_TRITON_SOFTMAXV", "0") != "1":
        return None
    try:
        return TritonSoftmaxV()
    except Exception:
        return None
