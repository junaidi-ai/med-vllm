import os
import json
import time
from dataclasses import asdict, dataclass
from typing import List

import torch

from medvllm.optim.fusion import get_fused_bias_gelu


@dataclass
class Case:
    B: int
    N: int
    H: int
    iters: int = 100


def run_case(case: Case):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(
        case.B * case.N,
        case.H,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    bias = torch.randn(case.H, device=device, dtype=x.dtype)

    gelu = torch.nn.GELU(approximate="tanh").to(device)

    # Eager baseline: y = gelu(x + b)
    def eager_step(inp):
        return gelu(inp + bias)

    # Triton fused
    os.environ.setdefault("MEDVLLM_ENABLE_TRITON_BIAS_GELU", "1")
    fused = get_fused_bias_gelu(case.H, bias=True)
    if fused is not None:
        with torch.no_grad():
            fused.bias.copy_(bias.to(torch.float32))

    # Warmup
    for _ in range(10):
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

    eager_time = t1 - t0
    tokens = case.B * case.N

    res = {
        "case": asdict(case),
        "device": device.type,
        "dtype": str(x.dtype).replace("torch.", ""),
        "results": [
            {
                "impl": "eager_bias_gelu",
                "elapsed_s": eager_time,
                "tokens": tokens,
                "tokens_per_s": tokens * case.iters / max(eager_time, 1e-9),
            }
        ],
    }
    if fused_time is not None:
        res["results"].append(
            {
                "impl": "triton_bias_gelu",
                "elapsed_s": fused_time,
                "tokens": tokens,
                "tokens_per_s": tokens * case.iters / max(fused_time, 1e-9),
            }
        )
    return res


def main():
    cases: List[Case] = [
        Case(B=8, N=8, H=512, iters=200),
        Case(B=4, N=16, H=1024, iters=200),
        Case(B=2, N=32, H=2048, iters=200),
    ]
    out = {
        "suite": "bias_gelu",
        "cases": [run_case(c) for c in cases],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
