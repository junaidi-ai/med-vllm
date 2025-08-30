import os
import pytest
import torch

from medvllm.optim.fusion import get_fused_separable_conv3d


def _have_triton_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_triton_cuda(), reason="CUDA+Triton required")
@pytest.mark.parametrize(
    "B,C,D,H,W",
    [
        (1, 4, 8, 16, 16),
        (2, 8, 8, 32, 32),
    ],
)
def test_sep3d_depthwise_parity(B, C, D, H, W):
    os.environ["MEDVLLM_ENABLE_TRITON_SEP3D"] = "1"

    device = torch.device("cuda")
    dtype = torch.float32

    x = torch.randn(B, C, D, H, W, device=device, dtype=dtype)
    weight = torch.randn(C, 1, 3, 3, 3, device=device, dtype=dtype)
    bias = torch.randn(C, device=device, dtype=dtype)

    # eager depthwise
    y_ref = torch.nn.functional.conv3d(x, weight, bias=bias, stride=1, padding=1, groups=C)

    fused = get_fused_separable_conv3d(C, bias=True)
    assert fused is not None, "Expected Triton separable conv3d available when enabled"

    # Load params
    with torch.no_grad():
        fused.weight.copy_(weight.squeeze(1))
        fused.bias.copy_(bias)

    y_fused = fused(x)

    atol = 1e-5
    rtol = 1e-5
    assert torch.allclose(
        y_fused, y_ref, atol=atol, rtol=rtol
    ), f"Mismatch: max abs={(y_fused - y_ref).abs().max().item()}"
