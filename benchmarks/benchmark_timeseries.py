"""
Time-series micro-benchmark with unified profiler.

Examples:
  # Synthetic CSV auto-generated
  python benchmarks/benchmark_timeseries.py --model gru --device cpu --iters 50

  # Use your CSV
  python benchmarks/benchmark_timeseries.py --csv path/to/series.csv --features f0 f1 f2 --model conv1d --device cuda
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from medvllm.data import get_dataset, MedicalDatasetConfig
from medvllm.utils.profiler import get_profiler

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def make_synthetic_csv(path: Path, rows: int = 1024, features: int = 8) -> None:
    rng = np.random.default_rng(123)
    df = {
        "series_id": ["s1"] * rows,
        "t": np.arange(rows),
    }
    for i in range(features):
        # AR(1)-ish series
        x = rng.standard_normal(size=rows).astype(np.float32)
        for j in range(1, rows):
            x[j] = 0.8 * x[j - 1] + 0.2 * x[j]
        df[f"f{i}"] = x
    # target is next-step of f0
    y = np.roll(df["f0"], -1)
    y[-1] = y[-2]
    df["y"] = y
    import pandas as _pd

    _pd.DataFrame(df).to_csv(path, index=False)


class GRUHead(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden, num_layers=num_layers, batch_first=True
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.gru(x)
        last = out[:, -1, :]
        y = self.head(last)[:, 0]
        return y


class Conv1DHead(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64, kernel: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden, kernel_size=kernel, padding=kernel // 2)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        h = self.conv(x)
        h = self.act(h)
        h = self.pool(h)  # [B, hidden, 1]
        h = h[:, :, 0]
        y = self.head(h)[:, 0]
        return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--dir", type=str, default="")
    ap.add_argument("--features", nargs="*", default=[])
    ap.add_argument("--target", type=str, default="y")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model", type=str, choices=["gru", "conv1d"], default="gru")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--emit-trace", action="store_true")
    ap.add_argument("--trace-dir", type=str, default="./profiles")
    args = ap.parse_args()

    if pd is None and not args.csv and not args.dir:
        raise RuntimeError(
            "pandas is required for synthetic CSV generation. Install pandas or provide --csv/--dir."
        )

    csv_path: Path | None = None
    if not args.csv and not args.dir:
        tmpdir = Path(tempfile.mkdtemp(prefix="ts_bench_"))
        csv_path = tmpdir / "series.csv"
        make_synthetic_csv(csv_path)
    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

    feature_cols = args.features if args.features else None

    cfg = MedicalDatasetConfig(
        name="timeseries_bench",
        timeseries_path=str(csv_path) if csv_path else None,
        timeseries_dir=args.dir or None,
        file_pattern="*.csv",
        series_id_col="series_id",
        time_col="t",
        feature_cols=feature_cols,
        target_col=args.target,
        window=args.window,
        stride=args.stride,
        normalization="zscore",
        drop_last_incomplete=False,
        batch_size=args.batch,
        shuffle=False,
    )

    ds = get_dataset(cfg)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    # Determine feature size
    first = next(iter(dl))
    F = first["input"].shape[-1]

    device = torch.device(args.device)
    if args.model == "gru":
        model = GRUHead(F, hidden=args.hidden)
    else:
        model = Conv1DHead(F, hidden=args.hidden)
    model.to(device)
    loss_fn = nn.MSELoss()

    profiler = get_profiler(
        device=args.device, emit_trace=args.emit_trace, trace_dir=args.trace_dir
    )

    # Warmup one batch to avoid lazy init costs
    with torch.no_grad():
        xb = first["input"].to(device)
        model(xb)

    n_items = 0
    lat_ms = []
    with profiler.profile():
        it = iter(dl)
        for i in range(args.iters):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            xb = batch["input"].to(device)
            yb = batch["target"].to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                yp = model(xb)
                loss = loss_fn(yp, yb)
            t1 = time.perf_counter()
            lat_ms.append((t1 - t0) * 1000.0)
            n_items += xb.shape[0]

    results = {
        "device": args.device,
        "model": args.model,
        "iters": args.iters,
        "batch": args.batch,
        "window": args.window,
        "stride": args.stride,
        "features": F,
        "throughput_items_per_s": float(n_items / (sum(lat_ms) / 1000.0)),
        "latency_ms_per_batch_avg": float(sum(lat_ms) / len(lat_ms)) if lat_ms else 0.0,
        "profiler": getattr(profiler, "results", {}),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
