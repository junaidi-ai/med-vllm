import itertools
import json
import os
import statistics
from collections import defaultdict

import torch

from benchmark_separable_conv3d import Case, run_case

# Candidate configs to sweep
BLOCK_WS = [64, 128, 256]
WARPS = [2, 4, 8]
STAGES = [2, 3, 4]

# Shape buckets (representative)
BUCKETS = [
    Case(B=1, C=16, D=32, H=64, W=64, iters=40),
    Case(B=1, C=32, D=32, H=128, W=128, iters=30),
    Case(B=1, C=32, D=64, H=128, W=256, iters=20),
]


def run_sweep():
    if not torch.cuda.is_available():
        print(json.dumps({"error": "cuda_unavailable"}))
        return

    results = []
    for case in BUCKETS:
        case_key = f"B{case.B}_C{case.C}_D{case.D}_H{case.H}_W{case.W}"
        best = None
        for bw, nw, ns in itertools.product(BLOCK_WS, WARPS, STAGES):
            os.environ["MEDVLLM_ENABLE_TRITON_SEP3D"] = "1"
            os.environ["MEDVLLM_SEP3D_BLOCK_W"] = str(bw)
            os.environ["MEDVLLM_SEP3D_WARPS"] = str(nw)
            os.environ["MEDVLLM_SEP3D_STAGES"] = str(ns)
            # Repeat to reduce variance
            vps_samples = []
            for _ in range(3):
                res = run_case(case)
                tri = next((r for r in res["results"] if r["impl"] == "triton_dwconv3d"), None)
                if tri is None:
                    continue
                vps = float(tri["voxels_per_s"])
                # Filter clearly unrealistic outliers (e.g., due to timer underflow)
                if not (0.0 < vps < 1e12):
                    continue
                vps_samples.append(vps)
            if not vps_samples:
                continue
            vps = statistics.median(vps_samples)
            item = {"case": case_key, "bw": bw, "warps": nw, "stages": ns, "voxels_per_s": vps}
            results.append(item)
            if best is None or vps > best["voxels_per_s"]:
                best = item
        if best:
            print(f"Best for {case_key}: {best}")
    # Aggregate winners: choose top-N unique configs across buckets
    # Keep unique by (bw,warps,stages), sorted by mean vps over buckets where it appeared
    perf_by_cfg = defaultdict(list)
    for r in results:
        perf_by_cfg[(r["bw"], r["warps"], r["stages"])].append(r["voxels_per_s"])
    ranked = sorted(
        (
            {
                "bw": k[0],
                "warps": k[1],
                "stages": k[2],
                "mean_vps": statistics.mean(v),
                "n": len(v),
            }
            for k, v in perf_by_cfg.items()
        ),
        key=lambda x: (x["mean_vps"], x["n"]),
        reverse=True,
    )
    top = ranked[:6]
    out = {
        "buckets": [case.__dict__ for case in BUCKETS],
        "top_configs": top,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run_sweep()
