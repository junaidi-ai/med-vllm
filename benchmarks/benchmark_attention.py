#!/usr/bin/env python3
"""
Benchmark attention backends: flash vs sdpa vs manual.

- Measures throughput (tokens/s approx via iterations per second) and latency.
- Reports CPU/GPU memory using local tests.medical.memory_profiler if available.
- Best-effort: skips unavailable backends and continues.

Usage:
  # Single-shape
  python -m benchmarks.benchmark_attention \
    --seq 512 --heads 16 --dim 64 --iters 50 --device auto --dtype bf16 [--save results.json]

  # Preset suite
  python -m benchmarks.benchmark_attention --suite basic --device auto --dtype bf16 [--save]

Output: prints a JSON summary for each available backend. If --save is passed,
writes a JSON file (path or default under benchmark_results_cpu_smoke/).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Optional, List, Tuple
import os
from datetime import datetime

# Torch optional
try:
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for this benchmark: {e}")

# Flash Attention optional
FLASH_AVAILABLE = False
try:
    from flash_attn import flash_attn_varlen_func as _flash_varlen  # type: ignore
    from flash_attn import flash_attn_qkvpacked_func as _flash_qkv  # type: ignore

    FLASH_AVAILABLE = True
except Exception:
    FLASH_AVAILABLE = False

# Unified profiler utility (prefers torch.profiler, falls back to local MemoryProfiler or no-op)
from medvllm.utils.profiler import get_profiler
from medvllm.optim.fusion import enable_compiler_fusion, get_fused_ffn, get_fused_softmaxv

try:
    from medvllm.kernels.triton_fused_ffn import EagerFFN  # type: ignore
except Exception:  # pragma: no cover
    EagerFFN = None  # type: ignore
try:
    from medvllm.kernels.triton_softmaxv import SoftmaxV  # type: ignore
except Exception:  # pragma: no cover
    SoftmaxV = None  # type: ignore


def manual_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def sdpa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> Optional[torch.Tensor]:
    if not hasattr(F, "scaled_dot_product_attention"):
        return None
    try:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
        )
    except Exception:
        return None


def flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> Optional[torch.Tensor]:
    if not FLASH_AVAILABLE or not q.is_cuda:
        return None
    try:
        # Pack QKV into single tensor (B, S, 3, H, D)
        qkv = torch.stack([q, k, v], dim=2)
        # flash_attn_qkvpacked_func expects (B, S, 3, H, D)
        return _flash_qkv(qkv, causal=True, softmax_scale=scale)
    except Exception:
        return None


class _AttentionModule(torch.nn.Module):
    def __init__(self, backend: str):
        super().__init__()
        self.backend = backend

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
        if self.backend == "manual":
            return manual_attention(q, k, v, scale)
        if self.backend == "sdpa":
            return sdpa_attention(q, k, v, scale)
        if self.backend == "flash":
            return flash_attention(q, k, v, scale)
        raise ValueError(f"Unknown backend {self.backend}")


def benchmark_once(
    backend: str,
    B: int,
    S: int,
    H: int,
    D: int,
    device: str,
    dtype: torch.dtype,
    iters: int,
    warmup: int = 10,
    emit_trace: bool = False,
    trace_dir: Optional[str] = None,
    fusion_compile: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(0)
    scale = D**-0.5
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    fn = {
        "manual": manual_attention,
        "sdpa": sdpa_attention,
        "flash": flash_attention,
    }.get(backend)
    if fn is None:
        raise ValueError(f"Unknown backend: {backend}")

    # Optionally wrap in a tiny module and enable compiler fusion
    module = None
    if fusion_compile:
        try:
            module = _AttentionModule(backend).to(device)
            module = enable_compiler_fusion(module, mode="max-autotune")
        except Exception:
            module = None

    # Warmup
    for _ in range(max(0, warmup)):
        out = module(q, k, v, scale) if module is not None else fn(q, k, v, scale)
        if out is None:
            return {"backend": backend, "available": False}
    if device == "cuda":
        torch.cuda.synchronize()

    profiler = get_profiler(device=device, emit_trace=emit_trace, trace_dir=trace_dir)

    # Timed loop
    t0 = time.perf_counter()
    with profiler.profile() as _prof_ctx:
        for _ in range(iters):
            out = module(q, k, v, scale) if module is not None else fn(q, k, v, scale)
            if out is None:
                return {"backend": backend, "available": False}
        if device == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    tokens = B * S * iters
    elapsed = t1 - t0
    toks_per_s = tokens / elapsed if elapsed > 0 else float("inf")
    res: Dict[str, Any] = {
        "backend": backend,
        "available": True,
        "elapsed_s": elapsed,
        "tokens": tokens,
        "tokens_per_s": toks_per_s,
        "B": B,
        "S": S,
        "H": H,
        "D": D,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
    }
    if getattr(profiler, "results", None) is not None:
        res["profiler"] = profiler.results
    return res


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument(
        "--suite",
        type=str,
        default=None,
        choices=["basic", "extended"],
        help="Run a preset suite of shapes instead of a single shape.",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        nargs="?",
        help="Optional path to save JSON results. If provided without value, saves to default dir.",
    )
    p.add_argument(
        "--emit-trace",
        action="store_true",
        help="Emit Chrome trace using torch.profiler when available.",
    )
    p.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Directory to write Chrome traces (default: ./profiles)",
    )
    p.add_argument(
        "--fusion-compile",
        action="store_true",
        help="Enable compiler-driven fusion for the attention op wrapper module.",
    )
    p.add_argument(
        "--ffn-bench",
        action="store_true",
        help="Run a microbenchmark comparing eager FFN vs Triton fused FFN (if available).",
    )
    p.add_argument(
        "--ffn-intermediate",
        type=int,
        default=256,
        help="Intermediate size for FFN bench (hidden from --dim).",
    )
    p.add_argument(
        "--attn-softmaxv-bench",
        action="store_true",
        help="Run a microbenchmark comparing eager softmax×V vs fused Triton softmax×V (if available).",
    )
    p.add_argument(
        "--enable-triton-softmaxv",
        action="store_true",
        help="Enable Triton softmax×V path for this run (sets MEDVLLM_ENABLE_TRITON_SOFTMAXV=1).",
    )
    args = p.parse_args()

    want_device = args.device
    device = "cuda" if want_device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # Optionally enable Triton softmax×V for this process
    if bool(getattr(args, "enable_triton_softmaxv", False)):
        os.environ["MEDVLLM_ENABLE_TRITON_SOFTMAXV"] = "1"

    backends = ["manual", "sdpa", "flash"]

    def run_case(B: int, S: int, H: int, D: int, iters: int) -> Dict[str, Any]:
        case_results = []
        for be in backends:
            r = benchmark_once(
                backend=be,
                B=B,
                S=S,
                H=H,
                D=D,
                device=device,
                dtype=dtype,
                iters=iters,
                emit_trace=bool(args.emit_trace),
                trace_dir=args.trace_dir,
                fusion_compile=bool(args.fusion_compile),
            )
            case_results.append(r)
        return {"B": B, "S": S, "H": H, "D": D, "iters": iters, "results": case_results}

    suites: Dict[str, List[Tuple[int, int, int, int, int]]] = {
        "basic": [
            (1, 128, args.heads, args.dim, max(20, args.iters // 2)),
            (1, 512, args.heads, args.dim, args.iters),
            (1, 1024, args.heads, args.dim, args.iters),
        ],
        "extended": [
            (1, 128, args.heads, args.dim, max(20, args.iters // 2)),
            (1, 512, args.heads, args.dim, args.iters),
            (1, 1024, args.heads, args.dim, args.iters),
            (2, 512, args.heads, args.dim, args.iters),
            (4, 512, args.heads, args.dim, max(10, args.iters // 2)),
        ],
    }

    payload: Dict[str, Any] = {"device": device, "dtype": str(dtype).replace("torch.", "")}
    if args.suite:
        cases = suites[args.suite]
        payload["suite"] = args.suite
        payload["cases"] = [run_case(B, S, H, D, iters) for (B, S, H, D, iters) in cases]
    else:
        payload["suite"] = None
        payload["cases"] = [run_case(args.batch, args.seq, args.heads, args.dim, args.iters)]

    # Optional FFN microbenchmark
    if bool(args.ffn_bench):
        try:
            hidden = int(args.dim)
            inter = int(args.ffn_intermediate)
            B, S = int(args.batch), int(args.seq)
            x = torch.randn(B * S, hidden, device=device, dtype=dtype)
            results_ffn: List[Dict[str, Any]] = []

            def _bench(module: torch.nn.Module, iters: int = 50) -> Dict[str, Any]:
                module.eval().to(device)
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = module(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(iters):
                        _ = module(x)
                    if device == "cuda":
                        torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                tps = (B * S * iters) / elapsed if elapsed > 0 else float("inf")
                return {"elapsed_s": elapsed, "tokens_per_s": tps}

            # Eager
            eager_mod = (
                EagerFFN(hidden, inter)
                if EagerFFN is not None
                else torch.nn.Sequential(
                    torch.nn.Linear(hidden, inter), torch.nn.GELU(), torch.nn.Linear(inter, hidden)
                )
            )
            r_eager = _bench(eager_mod)
            r_eager.update({"impl": "eager", "available": True})
            results_ffn.append(r_eager)

            # Triton fused (placeholder)
            fused = get_fused_ffn(hidden, inter)
            if fused is not None:
                r_fused = _bench(fused)
                r_fused.update({"impl": "triton_fused", "available": True})
                results_ffn.append(r_fused)
            else:
                results_ffn.append({"impl": "triton_fused", "available": False})

            payload["ffn_bench"] = {
                "hidden": hidden,
                "intermediate": inter,
                "iters": int(args.iters),
                "results": results_ffn,
            }
        except Exception as e:
            payload["ffn_bench"] = {"error": str(e)}

    # Optional softmax×V microbenchmark
    if bool(args.attn_softmaxv_bench):
        try:
            hidden = int(args.dim)
            H, D = int(args.heads), int(args.dim)
            B, S = int(args.batch), int(args.seq)
            dtype = dtype
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)
            scale = D**-0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            def _bench_softmaxv(module, iters: int = 50) -> Dict[str, Any]:
                # module(scores, v) -> out
                if hasattr(module, "to"):
                    module = module.to(device)
                # warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = module(scores, v)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(iters):
                        _ = module(scores, v)
                    if device == "cuda":
                        torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                tps = (B * S * iters) / elapsed if elapsed > 0 else float("inf")
                return {"elapsed_s": elapsed, "tokens_per_s": tps}

            results_smv: List[Dict[str, Any]] = []

            # Eager baseline
            if SoftmaxV is not None:
                eager = SoftmaxV()
                r_eager = _bench_softmaxv(eager, iters=int(args.iters))
                r_eager.update({"impl": "eager", "available": True})
                results_smv.append(r_eager)
            else:
                # fallback inline op
                class _Eager(nn.Module):
                    def forward(self, s, v):
                        return torch.matmul(torch.softmax(s, dim=-1), v)

                eager = _Eager()
                r_eager = _bench_softmaxv(eager, iters=int(args.iters))
                r_eager.update({"impl": "eager", "available": True})
                results_smv.append(r_eager)

            # Triton fused (placeholder)
            fused = get_fused_softmaxv()
            if fused is not None:
                r_fused = _bench_softmaxv(fused, iters=int(args.iters))
                r_fused.update({"impl": "triton_softmaxv", "available": True})
                results_smv.append(r_fused)
            else:
                results_smv.append({"impl": "triton_softmaxv", "available": False})

            payload["attn_softmaxv_bench"] = {
                "hidden": hidden,
                "heads": H,
                "seq": S,
                "iters": int(args.iters),
                "results": results_smv,
            }
        except Exception as e:
            payload["attn_softmaxv_bench"] = {"error": str(e)}

    text = json.dumps(payload, indent=2)
    print(text)

    # Optional save
    if args.save is not None:
        out_path = args.save
        if out_path is None or out_path == "":
            # default path under benchmarks/
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("benchmarks/benchmark_results_cpu_smoke", exist_ok=True)
            out_path = os.path.join(
                "benchmarks/benchmark_results_cpu_smoke", f"attention_{device}_{ts}.json"
            )
        # If save provided as flag without value (via nargs="?"), args.save may be string "None"; guard
        if out_path == "None" or out_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("benchmarks/benchmark_results_cpu_smoke", exist_ok=True)
            out_path = os.path.join(
                "benchmarks/benchmark_results_cpu_smoke", f"attention_{device}_{ts}.json"
            )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    main()
