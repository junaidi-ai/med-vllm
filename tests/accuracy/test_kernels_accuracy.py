import json
import os
import sys
import pytest
import torch

HAVE_CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False

skip_cuda = pytest.mark.skipif(not HAVE_CUDA, reason="CUDA not available")
skip_triton = pytest.mark.skipif(not HAVE_TRITON, reason="Triton not available")


def _set_env(d: dict):
    for k, v in d.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


DTYPES = [torch.float32, torch.float16]
if hasattr(torch, "bfloat16"):
    DTYPES.append(torch.bfloat16)


@skip_cuda
@skip_triton
@pytest.mark.parametrize("dtype", DTYPES)
def test_depthwise2d_eager_vs_triton_close(dtype):
    # Import here to avoid side-effects when Triton isn't present
    from medvllm.kernels.triton_depthwise_conv2d import build_fused_depthwise_conv2d_if_available

    device = torch.device("cuda")
    B, C, H, W = 2, 16, 128, 128
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)
    weight = torch.randn(C, 3, 3, device=device, dtype=dtype)
    bias = torch.randn(C, device=device, dtype=dtype)

    mod = build_fused_depthwise_conv2d_if_available(C)
    if mod is None:
        pytest.skip("DepthwiseConv2D Triton module unavailable")

    # Eager grouped conv
    y_ref = torch.nn.functional.conv2d(
        x, weight.view(C, 1, 3, 3), bias=bias, stride=1, padding=1, groups=C
    ).to(torch.float32)

    # Triton (module expects float32 output, cast internally as needed)
    mod.weight.data.copy_(weight)
    mod.bias = torch.nn.Parameter(bias)
    y_fused = mod(x).to(torch.float32)

    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-4
    else:
        atol, rtol = 5e-3, 5e-3

    assert torch.allclose(y_fused, y_ref, atol=atol, rtol=rtol)


@skip_cuda
@skip_triton
@pytest.mark.parametrize("dtype", DTYPES)
def test_sep3d_eager_vs_triton_close(dtype):
    from medvllm.kernels.triton_separable_conv3d import DepthwiseConv3d3x3x3

    device = torch.device("cuda")
    B, C, D, H, W = 1, 8, 16, 64, 64
    x = torch.randn(B, C, D, H, W, device=device, dtype=dtype)
    weight = torch.randn(C, 3, 3, 3, device=device, dtype=dtype)
    bias = torch.randn(C, device=device, dtype=dtype)

    # Eager grouped conv3d reference
    y_ref = torch.nn.functional.conv3d(
        x, weight.view(C, 1, 3, 3, 3), bias=bias, stride=1, padding=1, groups=C
    ).to(torch.float32)

    # Triton module (opt-in flag)
    _set_env({"MEDVLLM_ENABLE_TRITON_SEP3D": "1"})
    mod = DepthwiseConv3d3x3x3(C, bias=True)
    mod = mod.to(device)
    mod.weight.data.copy_(weight)
    mod.bias = torch.nn.Parameter(bias)

    # Optionally exercise fused path too
    for fuse_bias, fuse_relu in [(0, 0), (1, 0), (1, 1)]:
        _set_env(
            {
                "MEDVLLM_SEP3D_FUSE_BIAS": str(fuse_bias),
                "MEDVLLM_SEP3D_FUSE_RELU": str(fuse_relu),
            }
        )
        y_fused = mod(x).to(torch.float32)
        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-4
        else:
            atol, rtol = 5e-3, 5e-3
        assert torch.allclose(y_fused, y_ref, atol=atol, rtol=rtol)


@skip_cuda
@pytest.mark.parametrize(
    "impl", ["auto", "manual"]
)  # keep small; flash/sdpa may be unavailable in CI
@pytest.mark.parametrize("dtype", DTYPES)
def test_attention_accuracy_stable(dtype, impl):
    from medvllm.layers.attention import Attention

    device = torch.device("cuda")
    B, H, T, D = 2, 4, 64, 64
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)

    attn = Attention(impl=impl).to(device)
    out1 = attn(q, k, v)
    out2 = attn(q, k, v)

    # Stability check: same module twice should be close
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-4
    else:
        atol, rtol = 5e-3, 5e-3
    assert torch.allclose(out1, out2, atol=atol, rtol=rtol)
