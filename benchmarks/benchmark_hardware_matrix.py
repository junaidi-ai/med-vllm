"""Cross-hardware/backends benchmark matrix for Med vLLM.

This script:
- Detects available devices/backends via medvllm.hardware.detect
- Runs a small, representative inference workload using BioBERT/ClinicalBERT adapters
- Measures latency and memory (via tests.medical.memory_profiler)
- Emits per-backend JSON and a summary CSV under reports/<date>/hardware_matrix/

Usage:
  python -m benchmarks.benchmark_hardware_matrix --models biobert clinicalbert --seq-length 256 --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Project-local imports
from medvllm.hardware.detect import list_available_backends, get_hardware_profile
from medvllm.models.adapters import BioBERTAdapter, ClinicalBERTAdapter
from tests.medical.memory_profiler import MemoryProfiler


def _load_model(name: str, device: torch.device, precision: str = "fp16"):
    if name == "biobert":
        mdl = BioBERTAdapter.from_pretrained("monologg/biobert_v1.1_pubmed")
    elif name == "clinicalbert":
        mdl = ClinicalBERTAdapter.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    else:
        raise ValueError(f"Unknown model: {name}")
    if precision == "fp16" and device.type == "cuda":
        mdl = mdl.half()
    mdl = mdl.to(device)
    mdl.enable_cache()
    return mdl


def run_once(model, device: torch.device, batch: int, seq: int) -> float:
    vocab_size = 30522
    try:
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            vocab_size = len(model.tokenizer)
        else:
            cfg = getattr(getattr(model, "model", None), "config", None)
            if cfg is not None and hasattr(cfg, "vocab_size"):
                vocab_size = int(cfg.vocab_size)
    except Exception:
        pass

    input_ids = torch.randint(
        100, max(101, vocab_size - 1), (batch, seq), device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)

    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if start is not None:
        torch.cuda.synchronize()
        start.record()
    else:
        import time

        t0 = time.perf_counter()

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    if end is not None:
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        return float(ms)
    else:
        import time

        ms = (time.perf_counter() - t0) * 1000.0
        return float(ms)


def _run_tensorrt_native(
    model,
    model_name: str,
    batch: int,
    seq: int,
    tmp_dir: Path,
    *,
    cache: bool = True,
    bf16: bool = False,
    fp8: bool = False,
    min_batch: Optional[int] = None,
    opt_batch: Optional[int] = None,
    max_batch: Optional[int] = None,
    min_seq: Optional[int] = None,
    opt_seq: Optional[int] = None,
    max_seq: Optional[int] = None,
) -> float:
    """Build and run a native TensorRT engine from ONNX for maximum performance.
    Requires: tensorrt and pycuda (or cuda-python).

    Supports persistent engine caching and wider optimization profiles.
    """
    try:
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401  # initializes CUDA context
        import pycuda.driver as cuda
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"Native TensorRT path unavailable: {e}")

    onnx_path = tmp_dir / "model.onnx"
    input_ids, attention_mask = _export_to_onnx(model, torch.device("cpu"), batch, seq, onnx_path)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            errs = ", ".join([parser.get_error(i).desc() for i in range(parser.num_errors)])
            raise RuntimeError(f"TRT ONNX parse failed: {errs}")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    except Exception:
        config.max_workspace_size = 1 << 30  # TRT < 8.5
    # Precision flags
    if builder.platform_has_fast_fp16:
        try:
            config.set_flag(trt.BuilderFlag.FP16)
        except Exception:
            config.set_flag(trt.BuilderFlag.FP16)
    # Optional BF16/FP8 when available and requested
    if bf16 and hasattr(trt.BuilderFlag, "BF16"):
        try:
            config.set_flag(trt.BuilderFlag.BF16)
        except Exception:
            pass
    if fp8 and hasattr(trt.BuilderFlag, "FP8"):
        try:
            config.set_flag(trt.BuilderFlag.FP8)
        except Exception:
            pass

    # Optimization profile for explicit batch model (supports wider ranges)
    profile = builder.create_optimization_profile()
    # Handle 1 or 2 inputs
    in0 = network.get_input(0)
    # Derive profile bounds
    mb = min_batch if min_batch is not None else max(1, batch)
    ob = opt_batch if opt_batch is not None else batch
    xb = max_batch if max_batch is not None else batch
    ms = min_seq if min_seq is not None else max(1, seq)
    os_ = opt_seq if opt_seq is not None else seq
    xs = max_seq if max_seq is not None else seq
    profile.set_shape(in0.name, min=(mb, ms), opt=(ob, os_), max=(xb, xs))
    if network.num_inputs >= 2:
        in1 = network.get_input(1)
        profile.set_shape(in1.name, min=(mb, ms), opt=(ob, os_), max=(xb, xs))
    config.add_optimization_profile(profile)

    # Engine cache key
    cache_dir = tmp_dir / "trt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_file(path: Path) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    onnx_hash = _hash_file(onnx_path)
    flags_sig = [
        "fp16" if builder.platform_has_fast_fp16 else "fp32",
        "bf16" if (bf16 and hasattr(trt.BuilderFlag, "BF16")) else "nobf16",
        "fp8" if (fp8 and hasattr(trt.BuilderFlag, "FP8")) else "nofp8",
    ]
    prof_sig = f"b{mb}-{ob}-{xb}_s{ms}-{os_}-{xs}"
    plan_name = f"{model_name}_{onnx_hash}_{prof_sig}_{'_'.join(flags_sig)}_trt{getattr(trt, '__version__', 'unk')}.plan"
    plan_path = cache_dir / plan_name

    engine = None
    runtime = None
    if cache and plan_path.exists():
        try:
            runtime = trt.Runtime(logger)
            with open(plan_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
        except Exception:
            engine = None

    if engine is None:
        # Build (serialized when possible) and cache
        try:
            build_serialized = getattr(builder, "build_serialized_network", None)
            if build_serialized is not None:
                plan_bytes = build_serialized(network, config)
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(plan_bytes)
                if cache:
                    try:
                        with open(plan_path, "wb") as f:
                            f.write(plan_bytes)
                    except Exception:
                        pass
            else:
                engine = builder.build_engine(network, config)
                if cache and engine is not None:
                    try:
                        ser = engine.serialize()
                        with open(plan_path, "wb") as f:
                            f.write(bytearray(ser))
                    except Exception:
                        pass
        except Exception as e:
            raise RuntimeError(f"Failed to build TensorRT engine: {e}")
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    context = engine.create_execution_context()

    # Set binding shapes
    binding_idxs = {}
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        binding_idxs[name] = i
    # Prepare host inputs
    ids_np = input_ids.cpu().numpy().astype(np.int32)
    mask_np = attention_mask.cpu().numpy().astype(np.int32)

    # Set shapes and allocate device buffers
    def alloc_buf(shape, dtype):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        dptr = cuda.mem_alloc(nbytes)
        return dptr, nbytes

    # Determine input bindings
    inputs = []
    if network.num_inputs >= 2 and engine.get_binding_name(1) in binding_idxs:
        names = [engine.get_binding_name(0), engine.get_binding_name(1)]
        shapes = [ids_np.shape, mask_np.shape]
        host_arrs = [ids_np, mask_np]
    else:
        names = [engine.get_binding_name(0)]
        shapes = [ids_np.shape]
        host_arrs = [ids_np]

    bindings = [None] * engine.num_bindings
    for nm, shp, arr in zip(names, shapes, host_arrs):
        bi = engine.get_binding_index(nm)
        context.set_binding_shape(bi, tuple(shp))
        dptr, _ = alloc_buf(shp, arr.dtype)
        cuda.memcpy_htod(dptr, arr)
        bindings[bi] = int(dptr)

    # Output binding (assume single output)
    out_idx = None
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            out_idx = i
            break
    if out_idx is None:
        raise RuntimeError("No output binding found in TRT engine")
    out_shape = tuple(context.get_binding_shape(out_idx))
    out_dtype = trt.nptype(engine.get_binding_dtype(out_idx))
    d_out, _ = alloc_buf(out_shape, out_dtype)
    bindings[out_idx] = int(d_out)

    stream = cuda.Stream()
    import time

    t0 = time.perf_counter()
    ok = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TRT execute_async_v2 failed")
    stream.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    return float(ms)


def _export_to_onnx(
    model, device: torch.device, batch: int, seq: int, onnx_path: Path
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    vocab_size = 30522
    try:
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            vocab_size = len(model.tokenizer)
        else:
            cfg = getattr(getattr(model, "model", None), "config", None)
            if cfg is not None and hasattr(cfg, "vocab_size"):
                vocab_size = int(cfg.vocab_size)
    except Exception:
        pass
    input_ids = torch.randint(
        100, max(101, vocab_size - 1), (batch, seq), device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)

    dummy_inputs = (input_ids, attention_mask)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}}
    try:
        torch.onnx.export(
            model,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")
    return input_ids, attention_mask


def _compute_parity_metrics(ref_logits: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
    import math

    a = ref_logits.reshape(-1)
    b = logits.reshape(-1)
    mse = torch.mean((a - b) ** 2).item()
    denom = torch.norm(a) * torch.norm(b)
    cosine = (torch.dot(a, b) / denom).item() if denom > 0 else float("nan")
    # Top-1 match on CLS token
    ref_top1 = torch.argmax(ref_logits[:, 0, :], dim=-1)
    top1 = torch.argmax(logits[:, 0, :], dim=-1)
    top1_match = float((ref_top1 == top1).float().mean().item())
    # Per-token KL divergence (average over batch*seq)
    eps = 1e-8
    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
    test_log_probs = torch.log_softmax(logits, dim=-1)
    ref_probs = torch.softmax(ref_logits, dim=-1)
    kl = (ref_probs * (ref_log_probs - test_log_probs)).sum(dim=-1).mean().item()
    return {"acc_mse": mse, "acc_cosine": cosine, "acc_top1_match": top1_match, "acc_kl": kl}


def _run_onnxruntime(
    model, device: torch.device, batch: int, seq: int, precision: str, tmp_dir: Path
) -> float:
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(f"onnxruntime not available: {e}")

    onnx_path = tmp_dir / "model.onnx"
    input_ids, attention_mask = _export_to_onnx(model, torch.device("cpu"), batch, seq, onnx_path)

    providers = ["CPUExecutionProvider"]
    if device.type == "cuda":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    # Build feed dict using actual model input names; handle 1- or 2-input variants
    import numpy as np

    ids_np = input_ids.cpu().numpy().astype(np.int64)
    mask_np = attention_mask.cpu().numpy().astype(np.int64)
    inps = sess.get_inputs()
    if len(inps) >= 2:
        feed = {inps[0].name: ids_np, inps[1].name: mask_np}
    elif len(inps) == 1:
        feed = {inps[0].name: ids_np}
    else:
        feed = {"input_ids": ids_np}

    import time

    t0 = time.perf_counter()
    _ = sess.run(["logits"], feed)
    ms = (time.perf_counter() - t0) * 1000.0
    return float(ms)


def _run_tensorrt_via_ort(model, batch: int, seq: int, tmp_dir: Path) -> float:
    """Use ONNX Runtime with TensorRT EP when available."""
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(f"onnxruntime not available: {e}")

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime TensorRT EP not available")

    onnx_path = tmp_dir / "model.onnx"
    input_ids, attention_mask = _export_to_onnx(model, torch.device("cpu"), batch, seq, onnx_path)

    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    import time

    t0 = time.perf_counter()
    _ = sess.run(
        ["logits"],
        {"input_ids": input_ids.cpu().numpy(), "attention_mask": attention_mask.cpu().numpy()},
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return float(ms)


def _run_openvino(model, batch: int, seq: int, tmp_dir: Path) -> float:
    try:
        import openvino.runtime as ov
    except Exception as e:
        raise RuntimeError(f"openvino not available: {e}")

    onnx_path = tmp_dir / "model.onnx"
    input_ids, attention_mask = _export_to_onnx(model, torch.device("cpu"), batch, seq, onnx_path)

    core = ov.Core()
    ov_model = core.read_model(model=str(onnx_path))
    compiled = core.compile_model(ov_model, device_name="CPU")
    infer = compiled.create_infer_request()

    import numpy as np
    import time

    # Resolve real input names from compiled model
    input_ports = compiled.inputs
    feed = {}
    ids_np = input_ids.cpu().numpy().astype(np.int64)
    mask_np = attention_mask.cpu().numpy().astype(np.int64)
    if len(input_ports) >= 2:
        name0 = (
            input_ports[0].any_name
            if hasattr(input_ports[0], "any_name")
            else input_ports[0].get_any_name()
        )
        name1 = (
            input_ports[1].any_name
            if hasattr(input_ports[1], "any_name")
            else input_ports[1].get_any_name()
        )
        feed[name0] = ids_np
        feed[name1] = mask_np
    elif len(input_ports) == 1:
        # Some exports fold the mask; single input remains
        single_name = (
            input_ports[0].any_name
            if hasattr(input_ports[0], "any_name")
            else input_ports[0].get_any_name()
        )
        feed = {single_name: ids_np}
    else:
        # Fallback to common names; try only input_ids first
        feed = {"input_ids": ids_np}

    t0 = time.perf_counter()
    _ = infer.infer(feed)
    ms = (time.perf_counter() - t0) * 1000.0
    return float(ms)


def _run_tpu_torch_xla(model_name: str, batch: int, seq: int, precision: str) -> float:
    try:
        import torch_xla.core.xla_model as xm
    except Exception as e:
        raise RuntimeError(f"torch_xla not available: {e}")

    device = xm.xla_device()
    model = _load_model(model_name, device, precision="fp32")  # keep fp32 for safety on TPU

    import time

    t0 = time.perf_counter()
    _ = run_once(model, device, batch, seq)
    ms = (time.perf_counter() - t0) * 1000.0
    return float(ms)


def _maybe_quantize_int8_cpu(model):
    try:
        from torch.quantization import quantize_dynamic

        return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception:
        return model


def _extract_logits(out) -> torch.Tensor:
    """Best-effort extraction of logits tensor from various model outputs."""
    if isinstance(out, dict):
        if "logits" in out:
            return out["logits"]
        if "last_hidden_state" in out:
            return out["last_hidden_state"]
    if hasattr(out, "logits"):
        return out.logits
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, (tuple, list)) and len(out) > 0:
        elem = out[0]
        if isinstance(elem, torch.Tensor):
            return elem
    if isinstance(out, torch.Tensor):
        return out
    raise ValueError("Could not extract logits from model output")


def _make_shared_inputs(
    batch: int, seq: int, vocab_size: int = 30522, seed: int = 1234
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        100, max(101, vocab_size - 1), (batch, seq), generator=g, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["biobert"], choices=["biobert", "clinicalbert"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"])
    ap.add_argument("--output-root", type=str, default="reports")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "torch", "onnxruntime", "openvino", "tensorrt", "tpu"],
    )
    ap.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 path on CPU when possible (dynamic quantization)",
    )
    ap.add_argument(
        "--acc-check",
        action="store_true",
        help="Compute accuracy parity vs torch CPU fp32 on shared inputs",
    )
    # TensorRT options
    ap.add_argument(
        "--no-trt-cache",
        action="store_true",
        help="Disable TensorRT plan caching (default: enabled)",
    )
    ap.add_argument(
        "--trt-bf16", action="store_true", help="Enable TensorRT BF16 build flag when supported"
    )
    ap.add_argument(
        "--trt-fp8", action="store_true", help="Enable TensorRT FP8 build flag when supported"
    )
    ap.add_argument(
        "--trt-min-batch",
        type=int,
        default=None,
        help="Min batch for TensorRT optimization profile",
    )
    ap.add_argument(
        "--trt-opt-batch",
        type=int,
        default=None,
        help="Opt batch for TensorRT optimization profile",
    )
    ap.add_argument(
        "--trt-max-batch",
        type=int,
        default=None,
        help="Max batch for TensorRT optimization profile",
    )
    ap.add_argument(
        "--trt-min-seq",
        type=int,
        default=None,
        help="Min seq length for TensorRT optimization profile",
    )
    ap.add_argument(
        "--trt-opt-seq",
        type=int,
        default=None,
        help="Opt seq length for TensorRT optimization profile",
    )
    ap.add_argument(
        "--trt-max-seq",
        type=int,
        default=None,
        help="Max seq length for TensorRT optimization profile",
    )
    args = ap.parse_args()

    date_dir = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(args.output_root) / date_dir / "hardware_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    hw_profile = get_hardware_profile()
    backends = list_available_backends()
    if args.backend != "auto":
        # Filter detected backends to the selected one when possible
        if args.backend == "torch":
            # Keep torch-capable devices (cpu/cuda/xpu/mps)
            backends = [b for b in backends if b["name"] in {"cpu", "cuda", "xpu", "mps"}]
        else:
            backends = [b for b in backends if b["name"] == args.backend] or backends

    summary_rows: List[Dict[str, Any]] = []

    for be in backends:
        dev_str = be["device"]
        name = be["name"]
        device = torch.device(dev_str)

        # Skip fp16 on non-supported devices
        precision = args.precision
        if not be["precisions"].get(precision, False):
            precision = "fp32"

        for model_name in args.models:
            try:
                model = _load_model(model_name, device, precision)
                # Optional INT8 dynamic quant on CPU
                if args.int8 and device.type == "cpu":
                    # Optionally consume edge config; currently only advisory
                    cfg_path = Path("configs/deployment/edge_cpu_int8.json")
                    if cfg_path.exists():
                        try:
                            with open(cfg_path, "r", encoding="utf-8") as f:
                                _edge_cfg = json.load(f)
                        except Exception:
                            _edge_cfg = None
                    model = _maybe_quantize_int8_cpu(model)
            except Exception as e:
                # Record failure entry
                row = {
                    "backend": name,
                    "device": dev_str,
                    "model": model_name,
                    "precision": precision,
                    "status": "load_error",
                    "error": str(e),
                }
                summary_rows.append(row)
                continue

            # Warmup
            for _ in range(max(0, args.warmup)):
                try:
                    _ = run_once(model, device, args.batch_size, args.seq_length)
                except Exception:
                    pass

            mem_prof = MemoryProfiler(device=device.type)
            latencies: List[float] = []
            acc_metrics: Dict[str, Any] = {}

            # Accuracy parity vs torch CPU fp32 reference
            if args.acc_check:
                try:
                    ref_device = torch.device("cpu")
                    ref_model = _load_model(model_name, ref_device, precision="fp32")
                    # Derive vocab size if available
                    vocab_size = 30522
                    cfg = getattr(getattr(ref_model, "model", None), "config", None)
                    if cfg is not None and hasattr(cfg, "vocab_size"):
                        vocab_size = int(cfg.vocab_size)
                    in_ids_cpu, attn_cpu = _make_shared_inputs(
                        args.batch_size, args.seq_length, vocab_size=vocab_size
                    )
                    with torch.no_grad():
                        ref_out = ref_model(input_ids=in_ids_cpu, attention_mask=attn_cpu)
                        ref_logits = _extract_logits(ref_out).detach().to(torch.float32).cpu()
                except Exception as e:
                    ref_logits = None
                    acc_metrics["acc_error"] = f"ref_failed: {e}"

            with mem_prof.profile():
                for _ in range(max(1, args.iters)):
                    # Dispatch per backend
                    if args.backend in ("auto", "torch") or name in {"cpu", "cuda", "xpu", "mps"}:
                        ms = run_once(model, device, args.batch_size, args.seq_length)
                        # Accuracy check
                        if (
                            args.acc_check
                            and 'acc_error' not in acc_metrics
                            and ref_logits is not None
                        ):
                            try:
                                in_ids = in_ids_cpu.to(device)
                                attn = attn_cpu.to(device)
                                with torch.no_grad():
                                    out = model(input_ids=in_ids, attention_mask=attn)
                                    logits = _extract_logits(out).detach().to(torch.float32).cpu()
                            except Exception as e:
                                acc_metrics["acc_error"] = f"torch_failed: {e}"
                            else:
                                acc_metrics.update(_compute_parity_metrics(ref_logits, logits))
                    elif args.backend == "onnxruntime" or name == "onnxruntime":
                        ms = _run_onnxruntime(
                            model, device, args.batch_size, args.seq_length, precision, out_dir
                        )
                        if (
                            args.acc_check
                            and 'acc_error' not in acc_metrics
                            and ref_logits is not None
                        ):
                            try:
                                import onnxruntime as ort

                                onnx_path = out_dir / "model.onnx"
                                _ = _export_to_onnx(
                                    model,
                                    torch.device("cpu"),
                                    args.batch_size,
                                    args.seq_length,
                                    onnx_path,
                                )
                                providers = ["CPUExecutionProvider"]
                                if (
                                    device.type == "cuda"
                                    and "CUDAExecutionProvider" in ort.get_available_providers()
                                ):
                                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                                sess = ort.InferenceSession(str(onnx_path), providers=providers)
                                inps = sess.get_inputs()
                                ids_np = in_ids_cpu.numpy().astype('int64')
                                attn_np = attn_cpu.numpy().astype('int64')
                                if len(inps) >= 2:
                                    feed = {inps[0].name: ids_np, inps[1].name: attn_np}
                                elif len(inps) == 1:
                                    feed = {inps[0].name: ids_np}
                                else:
                                    feed = {"input_ids": ids_np}
                                out_np = sess.run(["logits"], feed)[0]
                                logits = torch.from_numpy(out_np).to(torch.float32)
                            except Exception as e:
                                acc_metrics["acc_error"] = f"ort_failed: {e}"
                            else:
                                acc_metrics.update(_compute_parity_metrics(ref_logits, logits))
                    elif args.backend == "openvino" or name == "openvino":
                        ms = _run_openvino(model, args.batch_size, args.seq_length, out_dir)
                        if (
                            args.acc_check
                            and 'acc_error' not in acc_metrics
                            and ref_logits is not None
                        ):
                            try:
                                import openvino.runtime as ov

                                onnx_path = out_dir / "model.onnx"
                                _ = _export_to_onnx(
                                    model,
                                    torch.device("cpu"),
                                    args.batch_size,
                                    args.seq_length,
                                    onnx_path,
                                )
                                core = ov.Core()
                                ov_model = core.read_model(model=str(onnx_path))
                                compiled = core.compile_model(ov_model, device_name="CPU")
                                infer = compiled.create_infer_request()
                                input_ports = compiled.inputs
                                ids_np = in_ids_cpu.numpy().astype('int64')
                                attn_np = attn_cpu.numpy().astype('int64')
                                if len(input_ports) >= 2:
                                    name0 = (
                                        input_ports[0].any_name
                                        if hasattr(input_ports[0], "any_name")
                                        else input_ports[0].get_any_name()
                                    )
                                    name1 = (
                                        input_ports[1].any_name
                                        if hasattr(input_ports[1], "any_name")
                                        else input_ports[1].get_any_name()
                                    )
                                    feed = {name0: ids_np, name1: attn_np}
                                elif len(input_ports) == 1:
                                    single_name = (
                                        input_ports[0].any_name
                                        if hasattr(input_ports[0], "any_name")
                                        else input_ports[0].get_any_name()
                                    )
                                    feed = {single_name: ids_np}
                                else:
                                    feed = {"input_ids": ids_np}
                                out_dict = infer.infer(feed)
                                # take first output tensor
                                out_np = list(out_dict.values())[0]
                                logits = torch.from_numpy(out_np).to(torch.float32)
                            except Exception as e:
                                acc_metrics["acc_error"] = f"ov_failed: {e}"
                            else:
                                a = ref_logits.reshape(-1)
                                b = logits.reshape(-1)
                                mse = torch.mean((a - b) ** 2).item()
                                denom = torch.norm(a) * torch.norm(b)
                                cosine = (
                                    (torch.dot(a, b) / denom).item() if denom > 0 else float('nan')
                                )
                                ref_top1 = torch.argmax(ref_logits[:, 0, :], dim=-1)
                                top1 = torch.argmax(logits[:, 0, :], dim=-1)
                                top1_match = float((ref_top1 == top1).float().mean().item())
                                acc_metrics.update(
                                    {
                                        "acc_mse": mse,
                                        "acc_cosine": cosine,
                                        "acc_top1_match": top1_match,
                                    }
                                )
                    elif args.backend == "tensorrt" or name == "tensorrt":
                        # Try native TensorRT path first, fallback to ORT TRT EP
                        try:
                            ms = _run_tensorrt_native(
                                model,
                                model_name,
                                args.batch_size,
                                args.seq_length,
                                out_dir,
                                cache=not args.no_trt_cache,
                                bf16=args.trt_bf16,
                                fp8=args.trt_fp8,
                                min_batch=args.trt_min_batch,
                                opt_batch=args.trt_opt_batch,
                                max_batch=args.trt_max_batch,
                                min_seq=args.trt_min_seq,
                                opt_seq=args.trt_opt_seq,
                                max_seq=args.trt_max_seq,
                            )
                        except Exception:
                            ms = _run_tensorrt_via_ort(
                                model, args.batch_size, args.seq_length, out_dir
                            )
                        # Accuracy via ONNX Runtime (TRT/CUDA/CPU providers) on shared inputs
                        if (
                            args.acc_check
                            and 'acc_error' not in acc_metrics
                            and ref_logits is not None
                        ):
                            try:
                                import onnxruntime as ort

                                onnx_path = out_dir / "model.onnx"
                                _ = _export_to_onnx(
                                    model,
                                    torch.device("cpu"),
                                    args.batch_size,
                                    args.seq_length,
                                    onnx_path,
                                )
                                providers = [
                                    "TensorrtExecutionProvider",
                                    "CUDAExecutionProvider",
                                    "CPUExecutionProvider",
                                ]
                                sess = ort.InferenceSession(str(onnx_path), providers=providers)
                                inps = sess.get_inputs()
                                ids_np = in_ids_cpu.numpy().astype('int64')
                                attn_np = attn_cpu.numpy().astype('int64')
                                if len(inps) >= 2:
                                    feed = {inps[0].name: ids_np, inps[1].name: attn_np}
                                elif len(inps) == 1:
                                    feed = {inps[0].name: ids_np}
                                else:
                                    feed = {"input_ids": ids_np}
                                out_np = sess.run(["logits"], feed)[0]
                                logits = torch.from_numpy(out_np).to(torch.float32)
                            except Exception as e:
                                acc_metrics["acc_error"] = f"trt_acc_via_ort_failed: {e}"
                            else:
                                acc_metrics.update(_compute_parity_metrics(ref_logits, logits))
                    else:
                        # Fallback to torch
                        ms = run_once(model, device, args.batch_size, args.seq_length)
                    latencies.append(ms)

            import numpy as np

            avg_ms = float(np.mean(latencies)) if latencies else float("nan")
            tps = (
                (args.batch_size * args.seq_length) / (avg_ms / 1000.0)
                if avg_ms == avg_ms and avg_ms > 0
                else float("nan")
            )

            # Determine status for summaries
            status = "ok"
            if acc_metrics.get("acc_error"):
                status = "acc_error"
            result = {
                "timestamp": datetime.now().isoformat(),
                "hardware_profile": hw_profile,
                "backend": be,
                "model": model_name,
                "precision": precision,
                "batch_size": args.batch_size,
                "seq_length": args.seq_length,
                "avg_latency_ms": avg_ms,
                "tokens_per_second": tps,
                "memory": mem_prof.results,
                "status": status,
            }
            if acc_metrics:
                result.update(acc_metrics)
            summary_rows.append(
                {
                    "backend": name,
                    "device": dev_str,
                    "model": model_name,
                    "precision": precision,
                    "batch_size": args.batch_size,
                    "seq_length": args.seq_length,
                    "avg_latency_ms": avg_ms,
                    "tokens_per_second": tps,
                    "status": status,
                    **acc_metrics,
                }
            )

    # Write summary CSV
    csv_path = out_dir / "summary.csv"
    if summary_rows:
        cols = [
            "backend",
            "device",
            "model",
            "precision",
            "batch_size",
            "seq_length",
            "avg_latency_ms",
            "tokens_per_second",
            "acc_mse",
            "acc_cosine",
            "acc_top1_match",
            "acc_kl",
            "acc_error",
            "status",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary_rows)

    # Write summary JSON for completeness
    json_sum = out_dir / "summary.json"
    with open(json_sum, "w", encoding="utf-8") as f:
        json.dump({"runs": summary_rows, "hardware_profile": hw_profile}, f, indent=2)

    # Write a simple Markdown summary
    md_lines = [
        "# Hardware Matrix Summary",
        "",
        f"Date: {date_dir}",
        "",
        "| Backend | Device | Model | Precision | Batch | Seq | Avg Latency (ms) | Tokens/s | Acc MSE | Acc Cosine | Top1 Match | Acc KL | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary_rows:
        md_lines.append(
            f"| {r.get('backend','')} | {r.get('device','')} | {r.get('model','')} | {r.get('precision','')} | {r.get('batch_size','')} | {r.get('seq_length','')} | {r.get('avg_latency_ms','')} | {r.get('tokens_per_second','')} | {r.get('acc_mse','')} | {r.get('acc_cosine','')} | {r.get('acc_top1_match','')} | {r.get('acc_kl','')} | {r.get('status','')} |"
        )
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Wrote reports to {out_dir}")


if __name__ == "__main__":
    main()
