"""Fusion scaffold utilities.

This module provides optional, compiler-driven optimizations that can be
enabled for models without changing model code. By default, nothing is
modified. When enabled, we currently leverage torch.compile to fuse and
optimize execution. This keeps maintenance low while benefiting from
backend improvements.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch


def _try_torch_compile(module: torch.nn.Module, mode: str = "max-autotune") -> torch.nn.Module:
    try:
        compiled = torch.compile(module, mode=mode)  # type: ignore[attr-defined]
        return compiled
    except Exception:
        return module


def enable_compiler_fusion(
    model: torch.nn.Module, *, mode: str = "max-autotune"
) -> torch.nn.Module:
    """Enable compiler-driven fusion/optimizations on the whole model.

    Currently wraps the entire module in torch.compile when available.
    Returns the (potentially) compiled model. If compilation fails, the
    original model is returned unchanged.
    """
    return _try_torch_compile(model, mode=mode)


@contextmanager
def compiler_fusion_enabled(model: torch.nn.Module, *, mode: str = "max-autotune"):
    """Context manager that compiles the model on enter and yields it.

    Usage:
        with compiler_fusion_enabled(model) as fused_model:
            out = fused_model(**inputs)
    """
    fused = enable_compiler_fusion(model, mode=mode)
    try:
        yield fused
    finally:
        # No explicit teardown necessary; caller keeps reference to original model.
        pass


def get_fused_ffn(
    hidden_size: int, intermediate_size: int, *, bias: bool = True
) -> Optional[torch.nn.Module]:
    """Return a fused-FFN module if a Triton-based path is available; else None.

    This is a thin wrapper over medvllm.kernels.triton_fused_ffn.build_fused_ffn_if_available.
    Intentionally lazy-imported to keep import costs low if Triton is not present.
    """
    try:
        from medvllm.kernels.triton_fused_ffn import build_fused_ffn_if_available  # type: ignore

        return build_fused_ffn_if_available(hidden_size, intermediate_size, bias=bias)
    except Exception:
        return None


def get_fused_separable_conv3d(channels: int, *, bias: bool = False) -> Optional[torch.nn.Module]:
    """Return a depthwise 3D separable conv module if available; else None.

    Thin wrapper over medvllm.kernels.triton_separable_conv3d.build_fused_separable_conv3d_if_available,
    lazy-imported to avoid overhead when Triton/CUDA are absent.
    """
    try:
        from medvllm.kernels.triton_separable_conv3d import (
            build_fused_separable_conv3d_if_available,
        )  # type: ignore

        return build_fused_separable_conv3d_if_available(channels, bias=bias)
    except Exception:
        return None


def get_fused_bias_gelu(hidden_size: int, *, bias: bool = True) -> Optional[torch.nn.Module]:
    """Return a fused Bias+GELU module if a Triton-based path is available; else None.

    Thin wrapper over medvllm.kernels.triton_bias_gelu.build_fused_bias_gelu_if_available,
    lazy-imported to keep import cost low when Triton/CUDA are absent.
    """
    try:
        from medvllm.kernels.triton_bias_gelu import build_fused_bias_gelu_if_available  # type: ignore

        return build_fused_bias_gelu_if_available(hidden_size, bias=bias)
    except Exception:
        return None


def get_fused_depthwise_conv2d(
    C: int, K: int, *, stride: int = 1, padding: int = 1, dilation: int = 1
) -> Optional[torch.nn.Module]:
    """Return a fused depthwise conv2d module if available; else None.

    Thin wrapper over medvllm.kernels.triton_depthwise_conv2d.build_fused_depthwise_conv2d_if_available
    with lazy import to avoid overhead when Triton/CUDA are absent.
    """
    try:
        from medvllm.kernels.triton_depthwise_conv2d import (
            build_fused_depthwise_conv2d_if_available,
        )  # type: ignore

        return build_fused_depthwise_conv2d_if_available(
            C, K, stride=stride, padding=padding, dilation=dilation
        )
    except Exception:
        return None


def get_fused_softmaxv() -> Optional[torch.nn.Module]:
    """Return a fused softmax×V module if a Triton-based path is available; else None.

    Thin wrapper over medvllm.kernels.triton_softmaxv.build_fused_softmaxv_if_available,
    lazy-imported to keep import cost low when Triton/CUDA are absent.
    """
    try:
        from medvllm.kernels.triton_softmaxv import build_fused_softmaxv_if_available  # type: ignore

        return build_fused_softmaxv_if_available()
    except Exception:
        return None
