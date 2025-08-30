import os
import json
import time
from dataclasses import asdict, dataclass
from typing import List

import torch

from medvllm.optim.fusion import get_fused_separable_conv3d


@dataclass
class Case:
    B: int
    C: int
    D: int
    H: int
    W: int
    iters: int = 50


def run_case(case: Case):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(
        case.B,
        case.C,
        case.D,
        case.H,
        case.W,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    # Eager depthwise via groups=C
    weight = torch.randn(case.C, 1, 3, 3, 3, device=device, dtype=x.dtype)
    bias = torch.randn(case.C, device=device, dtype=x.dtype)

    def eager_step(inp):
        return torch.nn.functional.conv3d(
            inp, weight, bias=bias, stride=1, padding=1, groups=case.C
        )

    # Triton fused
    os.environ.setdefault("MEDVLLM_ENABLE_TRITON_SEP3D", "1")
    fused = get_fused_separable_conv3d(case.C, bias=True)
    if fused is not None:
        with torch.no_grad():
            # map torch weight [C,1,3,3,3] -> kernel expected [C,3,3,3]
            fused.weight.copy_(weight.squeeze(1).to(torch.float32))
            if fused.bias is not None:
                fused.bias.copy_(bias.to(torch.float32))

    # Warmup
    for _ in range(5):
        y0 = eager_step(x)
        if fused is not None:
            y1 = fused(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time eager
    t0 = time.time()
    for _ in range(case.iters):
        y0 = eager_step(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    # Time fused
    fused_time = None
    if fused is not None:
        t2 = time.time()
        for _ in range(case.iters):
            y1 = fused(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.time()
        fused_time = t3 - t2

    voxels = case.B * case.C * case.D * case.H * case.W
    eager_time = t1 - t0

    res = {
        "case": asdict(case),
        "device": device.type,
        "dtype": str(x.dtype).replace("torch.", ""),
        "results": [
            {
                "impl": "eager_dwconv3d",
                "elapsed_s": eager_time,
                "voxels": voxels,
                "voxels_per_s": voxels * case.iters / max(eager_time, 1e-9),
            }
        ],
    }
    if fused_time is not None:
        res["results"].append(
            {
                "impl": "triton_dwconv3d",
                "elapsed_s": fused_time,
                "voxels": voxels,
                "voxels_per_s": voxels * case.iters / max(fused_time, 1e-9),
            }
        )
    return res


def main():
    cases: List[Case] = [
        Case(B=1, C=16, D=32, H=64, W=64, iters=50),
        Case(B=1, C=32, D=32, H=128, W=128, iters=30),
    ]
    out = {
        "suite": "separable_conv3d",
        "cases": [run_case(c) for c in cases],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
