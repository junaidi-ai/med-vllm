import os
import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.skipif(
    os.getenv("MEDVLLM_DISABLE_TRITON", "0") == "1" or not torch.cuda.is_available(),
    reason="Requires CUDA and Triton enabled",
)


def _has_triton_kernel():
    try:
        from medvllm.kernels.triton_depthwise_conv2d import (
            build_fused_depthwise_conv2d_if_available,
        )  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_triton_kernel(), reason="Triton module import failed")
@pytest.mark.parametrize(
    "C,H,W,K,stride,pad,dil",
    [
        # Original odd-K basic cases
        (8, 64, 64, 3, 1, 1, 1),
        (16, 128, 128, 3, 1, 1, 1),
        (32, 128, 96, 3, 1, 1, 1),
        (8, 64, 64, 5, 1, 2, 1),
        # New coverage: even K
        (8, 64, 64, 4, 1, 2, 1),
        (16, 96, 80, 6, 1, 3, 1),
        # New coverage: stride > 1
        (8, 65, 67, 3, 2, 1, 1),
        (12, 128, 130, 4, 2, 1, 1),
        # New coverage: dilation > 1
        (8, 64, 64, 3, 1, 2, 2),
        (8, 64, 64, 4, 1, 3, 2),
    ],
)
def test_triton_depthwise_matches_eager(C, H, W, K, stride, pad, dil):
    from medvllm.kernels.triton_depthwise_conv2d import (
        build_fused_depthwise_conv2d_if_available,
        EagerDepthwiseConv2d,
    )

    torch.manual_seed(0)
    x = torch.randn(2, C, H, W, device="cuda", dtype=torch.float32)

    # Reference eager depthwise conv
    eager = EagerDepthwiseConv2d(C, K, stride=stride, padding=pad, dilation=dil, bias=False).to(
        "cuda"
    )

    # Fused triton conv (weights initialized randomly; copy eager weights for exact comparison)
    fused = build_fused_depthwise_conv2d_if_available(
        C, K, stride=stride, padding=pad, dilation=dil
    )
    assert fused is not None, "Expected Triton fused module to be available on CUDA"
    fused = fused.to("cuda")

    # Align weights so outputs are comparable
    with torch.no_grad():
        # eager.weight: [C,1,K,K] or [C, K, K] depending on Conv2d internals; retrieve correctly
        w_eager = eager.op.weight.detach().reshape(C, K, K).contiguous()
        fused.weight.copy_(w_eager)

    y_ref = eager(x)
    y_fused = fused(x)

    # Strict-ish tolerance for fp32
    torch.testing.assert_close(y_fused, y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_triton_kernel(), reason="Triton module import failed")
def test_triton_depthwise_dtype_support():
    from medvllm.kernels.triton_depthwise_conv2d import (
        build_fused_depthwise_conv2d_if_available,
        EagerDepthwiseConv2d,
    )

    dtypes = [torch.float32]
    if hasattr(torch, "bfloat16"):
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        C, H, W, K = 8, 64, 64, 3
        x = torch.randn(2, C, H, W, device="cuda", dtype=dtype)

        eager = EagerDepthwiseConv2d(C, K, bias=False).to("cuda").to(dtype)
        fused = build_fused_depthwise_conv2d_if_available(C, K)
        assert fused is not None
        fused = fused.to("cuda").to(dtype)

        with torch.no_grad():
            w_eager = eager.op.weight.detach().reshape(C, K, K).contiguous().to(dtype)
            fused.weight.copy_(w_eager)

        y_ref = eager(x)
        y_fused = fused(x)

        # Looser tolerance for bf16
        is_bf16 = hasattr(torch, "bfloat16") and dtype is torch.bfloat16
        rtol = 2e-2 if is_bf16 else 1e-4
        atol = 2e-2 if is_bf16 else 1e-4
        torch.testing.assert_close(y_fused, y_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not _has_triton_kernel(), reason="Triton module import failed")
@pytest.mark.parametrize(
    "C,H,W,K,stride,pad,dil",
    [
        (8, 64, 64, 3, 1, 1, 1),
        (8, 64, 64, 4, 1, 2, 1),
        (8, 65, 67, 3, 2, 1, 1),
        (8, 64, 64, 3, 1, 2, 2),
    ],
)
def test_triton_depthwise_channels_last(C, H, W, K, stride, pad, dil):
    from medvllm.kernels.triton_depthwise_conv2d import (
        build_fused_depthwise_conv2d_if_available,
        EagerDepthwiseConv2d,
    )

    torch.manual_seed(1)
    x = torch.randn(2, C, H, W, device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )

    eager = (
        EagerDepthwiseConv2d(C, K, stride=stride, padding=pad, dilation=dil, bias=False)
        .to("cuda")
        .to(memory_format=torch.channels_last)
    )
    fused = build_fused_depthwise_conv2d_if_available(
        C, K, stride=stride, padding=pad, dilation=dil
    )
    assert fused is not None
    fused = fused.to("cuda")

    with torch.no_grad():
        w_eager = eager.op.weight.detach().reshape(C, K, K).contiguous()
        fused.weight.copy_(w_eager)

    y_ref = eager(x)
    y_fused = fused(x)

    torch.testing.assert_close(y_fused, y_ref, rtol=1e-4, atol=1e-4)
