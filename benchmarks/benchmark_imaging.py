#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medvllm.utils.profiler import get_profiler
from medvllm.optim.fusion import enable_compiler_fusion
from medvllm.optim.fusion import get_fused_separable_conv3d


@dataclass
class BenchResult:
    device: str
    dtype: str
    channels_last: bool
    amp: bool
    cudnn_benchmark: bool
    compiled: bool
    input_shape: tuple
    batches: int
    batch_time_ms: float
    imgs_per_sec: float
    cpu_max_rss_mb: Optional[float]
    cuda_max_mem_mb: Optional[float]
    # Optional accuracy/repeatability checks
    acc_check_enabled: Optional[bool] = None
    has_nan: Optional[bool] = None
    has_inf: Optional[bool] = None
    repeatability_pass: Optional[bool] = None
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None


class TinyConvNet2D(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, groups: int = 1):
        super().__init__()
        g1 = max(1, min(groups, in_ch))
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1, groups=g1)
        g2 = max(1, min(groups, 16))
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, groups=g2)
        g3 = max(1, min(groups, 32))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, groups=g3)
        self.pool = nn.MaxPool2d(2)
        # Resolution-agnostic head: adapt to fixed 4x4 before FC
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
        # Lightweight segmentation head for toy 2D segmentation
        self.seg_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.pool(x)
        x = F.silu(self.conv2(x))
        x = self.pool(x)
        x = F.silu(self.conv3(x))
        x = self.pool(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class TinyDepthwiseNet3D(nn.Module):
    """True depthwise 3D network for apples-to-apples comparisons.

    Each stage:
      - Depthwise 3D conv: groups=in_channels, kernel 3, padding 1
        Prefer fused separable depthwise conv3d when available via
        medvllm.optim.fusion.get_fused_separable_conv3d.
      - Pointwise 1x1x1 conv to change channels (standard conv, groups=1)
      - SiLU + MaxPool3d(2)
    Final head matches TinyConvNet3D for fair comparison.
    """

    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        # Stage 1: in_ch -> 16
        self.dw1 = get_fused_separable_conv3d(in_ch, bias=True) or nn.Conv3d(
            in_ch, in_ch, 3, padding=1, groups=in_ch, bias=True
        )
        self.pw1 = nn.Conv3d(in_ch, 16, kernel_size=1)
        # Stage 2: 16 -> 32
        self.dw2 = get_fused_separable_conv3d(16, bias=True) or nn.Conv3d(
            16, 16, 3, padding=1, groups=16, bias=True
        )
        self.pw2 = nn.Conv3d(16, 32, kernel_size=1)
        # Stage 3: 32 -> 64
        self.dw3 = get_fused_separable_conv3d(32, bias=True) or nn.Conv3d(
            32, 32, 3, padding=1, groups=32, bias=True
        )
        self.pw3 = nn.Conv3d(32, 64, kernel_size=1)
        self.pool = nn.MaxPool3d(2)
        self.adapt = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc = nn.Linear(64 * 4 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.silu(self.pw1(self.dw1(x)))
        x = self.pool(x)
        x = F.silu(self.pw2(self.dw2(x)))
        x = self.pool(x)
        x = F.silu(self.pw3(self.dw3(x)))
        x = self.pool(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def forward_features(self, x):
        """Return feature map before adaptive pooling for toy segmentation."""
        x = F.silu(self.conv1(x))
        x = self.pool(x)
        x = F.silu(self.conv2(x))
        x = self.pool(x)
        x = F.silu(self.conv3(x))
        x = self.pool(x)
        return x


class TinyConvNet3D(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, groups: int = 1):
        super().__init__()
        g1 = max(1, min(groups, in_ch))
        self.conv1 = nn.Conv3d(in_ch, 16, 3, padding=1, groups=g1)
        g2 = max(1, min(groups, 16))
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1, groups=g2)
        g3 = max(1, min(groups, 32))
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1, groups=g3)
        self.pool = nn.MaxPool3d(2)
        self.adapt = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc = nn.Linear(64 * 4 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.pool(x)
        x = F.silu(self.conv2(x))
        x = self.pool(x)
        x = F.silu(self.conv3(x))
        x = self.pool(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_memory(device: str) -> Dict[str, Optional[float]]:
    cpu_mb = None
    cuda_mb = None
    # psutil optional
    try:
        import psutil  # type: ignore

        cpu_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except Exception:
        pass
    if device == "cuda" and torch.cuda.is_available():
        try:
            cuda_mb = torch.cuda.max_memory_allocated() / (1024**2)
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    return {"cpu_max_rss_mb": cpu_mb, "cuda_max_mem_mb": cuda_mb}


def main():
    p = argparse.ArgumentParser(description="Simple imaging conv benchmark")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=32, help="Depth for 3D inputs")
    p.add_argument("--in-ch", type=int, default=1, help="Input channels")
    p.add_argument(
        "--groups", type=int, default=1, help="Grouping for conv layers (use in_ch for depthwise)"
    )
    p.add_argument(
        "--depthwise", action="store_true", help="Enable depthwise conv (sets groups=in_ch)"
    )
    p.add_argument(
        "--conv-type",
        type=str,
        default="2d",
        choices=["2d", "3d"],
        help="Convolution dimensionality",
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cudnn-benchmark", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--fusion-compile",
        action="store_true",
        help="Enable compiler-driven fusion via medvllm.optim.fusion.enable_compiler_fusion().",
    )
    p.add_argument(
        "--out", type=str, default="benchmarks/benchmark_results_cpu_smoke/conv_bench.json"
    )
    p.add_argument(
        "--acc-check",
        action="store_true",
        help="Run basic repeatability/numerical checks and include metrics in JSON.",
    )
    p.add_argument(
        "--acc-toy",
        action="store_true",
        help="Run toy classification accuracy/AUC evaluation using model logits vs random labels.",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes for toy classification accuracy (only used when --acc-toy).",
    )
    p.add_argument(
        "--seg-toy",
        action="store_true",
        help="Run toy 2D segmentation metrics (Dice/IoU) vs synthetic masks.",
    )
    p.add_argument(
        "--seg-threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing segmentation probabilities.",
    )
    p.add_argument(
        "--seg-fixture-dir",
        type=str,
        default=None,
        help="Directory with small real 2D segmentation fixtures to compute Dice/IoU. Expect *_img.(pt|npy) and *_mask.(pt|npy) pairs.",
    )
    p.add_argument(
        "--seg-dataset",
        type=str,
        default=None,
        choices=["seg2d_small"],
        help="Name of tiny real dataset to evaluate Dice/IoU (downloaded or generated deterministically).",
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
        "--depthwise-bench",
        action="store_true",
        help="Run microbenchmark for depthwise conv2d (2D only): eager vs fused.",
    )
    p.add_argument(
        "--depthwise-bench-iters",
        type=int,
        default=50,
        help="Iterations per case for depthwise microbenchmark.",
    )
    p.add_argument(
        "--depthwise-bench-sizes",
        type=str,
        default="",
        help="Comma-separated list of CxHxW cases for depthwise bench (e.g., '8x128x128,32x256x256'). Empty uses current --in-ch/--height/--width.",
    )
    args = p.parse_args()

    # Resolve device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dtype
    dtype = args.dtype.lower()
    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]

    if args.cudnn_benchmark and torch.backends.cudnn.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    # Model
    groups = args.groups
    if args.depthwise:
        groups = max(1, args.in_ch)
    if args.conv_type == "3d":
        if args.depthwise:
            # True depthwise 3D path using depthwise+pointwise blocks
            model = TinyDepthwiseNet3D(in_ch=args.in_ch, num_classes=max(2, int(args.num_classes)))
        else:
            model = TinyConvNet3D(
                in_ch=args.in_ch, num_classes=max(2, int(args.num_classes)), groups=groups
            )
    else:
        model = TinyConvNet2D(
            in_ch=args.in_ch, num_classes=max(2, int(args.num_classes)), groups=groups
        )
    model.eval()
    model.to(device)

    if args.compile:
        try:
            model = torch.compile(model, mode="max-autotune")  # type: ignore[attr-defined]
        except Exception:
            pass
    if args.fusion_compile:
        try:
            model = enable_compiler_fusion(model, mode="max-autotune")
        except Exception:
            pass

    # Input
    if args.conv_type == "3d":
        shape = (args.batch, args.in_ch, args.depth, args.height, args.width)
    else:
        shape = (args.batch, args.in_ch, args.height, args.width)
    x = torch.randn(shape, device=device)
    if args.channels_last and device == "cuda":
        if args.conv_type == "3d":
            x = x.to(memory_format=torch.channels_last_3d)
            model = model.to(memory_format=torch.channels_last_3d)
        else:
            x = x.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

    # AMP context
    use_amp = not args.no_amp and device == "cuda" and dtype in ("fp16", "bf16")

    # Measure with unified profiler
    profiler = get_profiler(
        device=device, emit_trace=bool(args.emit_trace), trace_dir=args.trace_dir
    )
    start = time.perf_counter()
    with profiler.profile():
        with torch.no_grad():
            if use_amp:
                autocast_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    for _ in range(args.batches):
                        _ = model(x)
            else:
                for _ in range(args.batches):
                    _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    imgs = args.batch * args.batches
    ips = imgs / (elapsed_ms / 1000.0)
    # Prefer profiler memory results if present
    prof_res = getattr(profiler, "results", None) or {}
    mem = {
        "cpu_max_rss_mb": prof_res.get("cpu_max_rss_mb", None),
        "cuda_max_mem_mb": prof_res.get("cuda_max_mem_mb", None),
    }
    if mem["cpu_max_rss_mb"] is None and mem["cuda_max_mem_mb"] is None:
        mem = get_memory(device)

    # Optional accuracy / repeatability checks (deterministic input, two runs)
    acc_enabled = bool(args.acc_check)
    has_nan = None
    has_inf = None
    repeatability_pass = None
    max_abs_diff = None
    mean_abs_diff = None
    if acc_enabled:
        torch.manual_seed(0)
        x_check = torch.randn(shape, device=device)
        if args.channels_last and device == "cuda":
            x_check = x_check.to(memory_format=torch.channels_last)
        with torch.no_grad():
            if use_amp:
                autocast_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    y1 = model(x_check)
                    y2 = model(x_check)
            else:
                y1 = model(x_check)
                y2 = model(x_check)
        # Numerical stats
        has_nan = bool(torch.isnan(y1).any().item() or torch.isnan(y2).any().item())
        has_inf = bool(torch.isinf(y1).any().item() or torch.isinf(y2).any().item())
        diff = (y1 - y2).abs()
        max_abs_diff = float(diff.max().item())
        mean_abs_diff = float(diff.mean().item())
        # Tolerance depends on amp/device
        tol = 1e-4
        if device == "cuda" and use_amp:
            tol = 5e-2
        repeatability_pass = (not has_nan) and (not has_inf) and (max_abs_diff <= tol)

    # Optional toy classification accuracy/AUC (separate from timing)
    toy_metrics: Optional[Dict[str, Any]] = None
    if bool(args.acc_toy):
        try:
            from medvllm.utils.metrics import compute_classification_metrics

            with torch.no_grad():
                # Single evaluation batch
                x_eval = torch.randn(shape, device=device)
                if args.channels_last and device == "cuda":
                    x_eval = x_eval.to(
                        memory_format=(
                            torch.channels_last_3d
                            if args.conv_type == "3d"
                            else torch.channels_last
                        )
                    )
                logits = model(x_eval)
                probs = torch.softmax(logits, dim=-1)
                # Random labels for toy eval
                num_classes = probs.shape[-1]
                y_true = torch.randint(
                    low=0, high=num_classes, size=(probs.shape[0],), device=probs.device
                )
                y_pred = torch.argmax(probs, dim=-1)
                # Move to CPU lists
                y_true_l = y_true.tolist()
                y_pred_l = y_pred.tolist()
                y_score_l = probs.detach().cpu().tolist()
                toy_metrics = compute_classification_metrics(
                    y_true_l, y_pred_l, average="macro", y_score=y_score_l
                )
        except Exception as e:
            toy_metrics = {"error": f"toy_metrics_failed: {e}"}

    # Optional toy 2D segmentation Dice/IoU (separate from timing)
    toy_segmentation: Optional[Dict[str, Any]] = None
    if bool(args.seg_toy) and args.conv_type == "2d":
        try:
            with torch.no_grad():
                x_eval = torch.randn(shape, device=device)
                if args.channels_last and device == "cuda":
                    x_eval = x_eval.to(memory_format=torch.channels_last)
                # Extract features and predict mask logits
                feats = model.forward_features(x_eval)  # type: ignore[attr-defined]
                # Upsample logits to input size
                seg_logits = model.seg_head(feats)
                seg_logits = F.interpolate(
                    seg_logits, size=(shape[-2], shape[-1]), mode="bilinear", align_corners=False
                )
                seg_probs = torch.sigmoid(seg_logits)
                # Synthetic ground truth masks
                y_true = (torch.rand_like(seg_probs) > 0.5).float()
                y_pred = (seg_probs >= float(args.seg_threshold)).float()
                # Dice and IoU
                eps = 1e-6
                inter = (y_pred * y_true).sum(dim=(1, 2, 3))
                pred_sum = y_pred.sum(dim=(1, 2, 3))
                true_sum = y_true.sum(dim=(1, 2, 3))
                dice = (2.0 * inter + eps) / (pred_sum + true_sum + eps)
                union = (y_pred + y_true - y_pred * y_true).sum(dim=(1, 2, 3))
                iou = (inter + eps) / (union + eps)
                toy_segmentation = {
                    "dice": float(dice.mean().item()),
                    "iou": float(iou.mean().item()),
                }
        except Exception as e:
            toy_segmentation = {"error": f"toy_segmentation_failed: {e}"}

    # Optional real 2D segmentation fixture evaluation
    seg_fixture: Optional[Dict[str, Any]] = None
    if args.seg_fixture_dir and args.conv_type == "2d":
        try:
            import numpy as np  # type: ignore
            from pathlib import Path

            fx_dir = Path(args.seg_fixture_dir)
            if fx_dir.is_dir():
                pairs = []
                # Match *_img.pt with *_mask.pt and *_img.npy with *_mask.npy
                img_pt = list(fx_dir.glob("*_img.pt"))
                mask_pt = {p.name.replace("_mask.pt", ""): p for p in fx_dir.glob("*_mask.pt")}
                for ip in img_pt:
                    key = ip.name.replace("_img.pt", "")
                    mp = mask_pt.get(key)
                    if mp:
                        pairs.append((ip, mp))
                img_npy = list(fx_dir.glob("*_img.npy"))
                mask_npy = {p.name.replace("_mask.npy", ""): p for p in fx_dir.glob("*_mask.npy")}
                for ip in img_npy:
                    key = ip.name.replace("_img.npy", "")
                    mp = mask_npy.get(key)
                    if mp:
                        pairs.append((ip, mp))

                dices = []
                ious = []
                with torch.no_grad():
                    for ip, mp in pairs:
                        if str(ip).endswith(".pt"):
                            img = torch.load(ip, map_location=device)
                            msk = torch.load(mp, map_location=device)
                        else:
                            img = torch.tensor(np.load(ip), device=device)
                            msk = torch.tensor(np.load(mp), device=device)

                        # Ensure shapes [H,W] or [C,H,W]; convert to [1,C,H,W]
                        if img.dim() == 2:
                            img = img.unsqueeze(0)
                        if img.dim() == 3 and img.shape[0] != args.in_ch:
                            img = img.repeat(max(1, args.in_ch // img.shape[0]), 1, 1)[: args.in_ch]
                        img = img.unsqueeze(0)  # [1,C,H,W]
                        if msk.dim() == 3 and msk.shape[0] == 1:
                            msk = msk.squeeze(0)
                        if msk.dim() == 2:
                            msk = msk.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                        elif msk.dim() == 3:
                            msk = msk.unsqueeze(0)
                        msk = (msk > 0.5).float()

                        feats = model.forward_features(img)
                        logits = model.seg_head(feats)
                        logits = F.interpolate(
                            logits, size=msk.shape[-2:], mode="bilinear", align_corners=False
                        )
                        probs = torch.sigmoid(logits)
                        pred = (probs >= float(args.seg_threshold)).float()

                        eps = 1e-6
                        inter = (pred * msk).sum(dim=(1, 2, 3))
                        pred_sum = pred.sum(dim=(1, 2, 3))
                        true_sum = msk.sum(dim=(1, 2, 3))
                        dice = (2.0 * inter + eps) / (pred_sum + true_sum + eps)
                        union = (pred + msk - pred * msk).sum(dim=(1, 2, 3))
                        iou = (inter + eps) / (union + eps)
                        dices.append(dice.mean().item())
                        ious.append(iou.mean().item())

                if dices:
                    seg_fixture = {
                        "count": len(dices),
                        "dice": float(sum(dices) / len(dices)),
                        "iou": float(sum(ious) / len(ious)),
                    }
                else:
                    seg_fixture = {"count": 0}
            else:
                seg_fixture = {"error": f"fixture_dir_not_found: {fx_dir}"}
        except Exception as e:
            seg_fixture = {"error": f"seg_fixture_failed: {e}"}

    # Optional dataset-based segmentation evaluation
    seg_dataset: Optional[Dict[str, Any]] = None
    if args.seg_dataset and args.conv_type == "2d":
        try:
            if args.seg_dataset == "seg2d_small":
                try:
                    from benchmarks.imaging_datasets import load_seg2d_small  # type: ignore
                except Exception:
                    # Ensure project root is on sys.path then retry
                    import sys
                    from pathlib import Path as _P

                    repo_root = _P(__file__).resolve().parents[1]
                    if str(repo_root) not in sys.path:
                        sys.path.insert(0, str(repo_root))
                    from benchmarks.imaging_datasets import load_seg2d_small  # type: ignore
                imgs, masks = load_seg2d_small()
            else:
                raise ValueError(f"Unknown seg dataset: {args.seg_dataset}")

            imgs = imgs.to(device)
            masks = masks.to(device).float()
            dices = []
            ious = []
            with torch.no_grad():
                for i in range(imgs.shape[0]):
                    img = imgs[i : i + 1]
                    msk = masks[i : i + 1]
                    feats = model.forward_features(img)
                    logits = model.seg_head(feats)
                    logits = F.interpolate(
                        logits, size=msk.shape[-2:], mode="bilinear", align_corners=False
                    )
                    probs = torch.sigmoid(logits)
                    pred = (probs >= float(args.seg_threshold)).float()
                    eps = 1e-6
                    inter = (pred * msk).sum(dim=(1, 2, 3))
                    pred_sum = pred.sum(dim=(1, 2, 3))
                    true_sum = msk.sum(dim=(1, 2, 3))
                    dice = (2.0 * inter + eps) / (pred_sum + true_sum + eps)
                    union = (pred + msk - pred * msk).sum(dim=(1, 2, 3))
                    iou = (inter + eps) / (union + eps)
                    dices.append(dice.mean().item())
                    ious.append(iou.mean().item())
            if dices:
                seg_dataset = {
                    "name": args.seg_dataset,
                    "count": len(dices),
                    "dice": float(sum(dices) / len(dices)),
                    "iou": float(sum(ious) / len(ious)),
                }
            else:
                seg_dataset = {"name": args.seg_dataset, "count": 0}
        except Exception as e:
            seg_dataset = {"name": args.seg_dataset, "error": f"seg_dataset_failed: {e}"}

    result = BenchResult(
        device=device,
        dtype=dtype,
        channels_last=bool(args.channels_last),
        amp=bool(use_amp),
        cudnn_benchmark=bool(args.cudnn_benchmark),
        compiled=bool(args.compile),
        input_shape=shape,
        batches=int(args.batches),
        batch_time_ms=float(elapsed_ms / args.batches),
        imgs_per_sec=float(ips),
        cpu_max_rss_mb=mem["cpu_max_rss_mb"],
        cuda_max_mem_mb=mem["cuda_max_mem_mb"],
        acc_check_enabled=acc_enabled,
        has_nan=has_nan,
        has_inf=has_inf,
        repeatability_pass=repeatability_pass,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = asdict(result)
    if 'toy_metrics' not in payload and 'toy_metrics' in locals() and toy_metrics is not None:
        payload["toy_metrics"] = toy_metrics
    if (
        'toy_segmentation' not in payload
        and 'toy_segmentation' in locals()
        and toy_segmentation is not None
    ):
        payload["toy_segmentation"] = toy_segmentation
    if 'seg_fixture' not in payload and 'seg_fixture' in locals() and seg_fixture is not None:
        payload["seg_fixture"] = seg_fixture
    if 'seg_dataset' not in payload and 'seg_dataset' in locals() and seg_dataset is not None:
        payload["seg_dataset"] = seg_dataset

    # Optional depthwise conv2d microbenchmark (2D only)
    if bool(getattr(args, "depthwise_bench", False)) and args.conv_type == "2d":
        try:
            # Lazy imports to avoid hard deps
            from medvllm.optim.fusion import get_fused_depthwise_conv2d  # type: ignore

            try:
                from medvllm.kernels.triton_depthwise_conv2d import EagerDepthwiseConv2d  # type: ignore
            except Exception:
                EagerDepthwiseConv2d = None  # type: ignore

            K = 3
            stride = 1
            padding = 1
            dilation = 1

            # Build case list
            cases = []
            if args.depthwise_bench_sizes.strip():
                for token in args.depthwise_bench_sizes.split(','):
                    token = token.strip()
                    if not token:
                        continue
                    parts = token.lower().split('x')
                    if len(parts) != 3:
                        continue
                    c_i, h_i, w_i = map(int, parts)
                    cases.append((c_i, h_i, w_i))
            else:
                cases.append((int(args.in_ch), int(args.height), int(args.width)))

            def _time_module(mod: nn.Module, x_bench: torch.Tensor, iters: int) -> Dict[str, Any]:
                mod.eval().to(device)
                with torch.no_grad():
                    for _ in range(5):
                        _ = mod(x_bench)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(iters):
                        _ = mod(x_bench)
                    if device == "cuda":
                        torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                imgs = x_bench.shape[0] * iters
                ips_local = imgs / elapsed if elapsed > 0 else float("inf")
                return {"elapsed_s": elapsed, "imgs_per_s": ips_local}

            bench_suite = {
                "K": K,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "iters": int(args.depthwise_bench_iters),
                "cases": [],
            }
            for C_i, H_i, W_i in cases:
                x_bench = torch.randn(
                    (max(1, int(args.batch)), C_i, max(8, H_i), max(8, W_i)),
                    device=device,
                    dtype=torch.float32,
                )

                # Eager reference
                if EagerDepthwiseConv2d is not None:
                    eager = EagerDepthwiseConv2d(
                        C_i, K, stride=stride, padding=padding, dilation=dilation, bias=False
                    )
                else:
                    eager = nn.Conv2d(
                        C_i,
                        C_i,
                        kernel_size=K,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=C_i,
                        bias=False,
                    )
                eager_res = _time_module(eager, x_bench, int(args.depthwise_bench_iters))

                # Fused (if available)
                fused = None
                if device == "cuda":
                    fused = get_fused_depthwise_conv2d(
                        C_i, K, stride=stride, padding=padding, dilation=dilation
                    )
                if (device == "cuda") and (fused is not None):
                    fused_res = {
                        **_time_module(fused, x_bench, int(args.depthwise_bench_iters)),
                        "available": True,
                    }
                else:
                    fused_res = {"available": False}

                bench_suite["cases"].append(
                    {
                        "C": C_i,
                        "H": H_i,
                        "W": W_i,
                        "eager": eager_res,
                        "fused": fused_res,
                    }
                )

            payload["depthwise_bench"] = bench_suite
        except Exception as e:
            payload["depthwise_bench"] = {"error": f"depthwise_bench_failed: {e}"}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
