import os
import math
import pytest
import torch

from medvllm.optim.fusion import get_fused_bias_gelu


def _have_triton_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_triton_cuda(), reason="CUDA+Triton required")
@pytest.mark.parametrize("B,N,H", [(2, 8, 512), (1, 16, 1024)])
def test_bias_gelu_parity(B, N, H):
    os.environ["MEDVLLM_ENABLE_TRITON_BIAS_GELU"] = "1"

    x = torch.randn(B * N, H, device="cuda", dtype=torch.float32)
    bias = torch.randn(H, device="cuda", dtype=torch.float32)

    fused = get_fused_bias_gelu(H, bias=True)
    assert fused is not None, "FusedBiasGELU should be available when enabled"
    # load bias
    with torch.no_grad():
        fused.bias.copy_(bias)

    y_fused = fused(x)

    gelu = torch.nn.GELU(approximate="tanh").cuda()
    y_ref = gelu(x + bias)

    atol = 1e-4
    rtol = 1e-4
    assert torch.allclose(
        y_fused, y_ref, atol=atol, rtol=rtol
    ), f"Mismatch: max abs={torch.max(torch.abs(y_fused - y_ref)).item()}"
