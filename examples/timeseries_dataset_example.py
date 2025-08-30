"""
Minimal example for TimeSeriesDataset.
- Generates a tiny synthetic CSV if no --csv is provided.
- Loads dataset via medvllm.data.get_dataset and iterates a DataLoader.
Run:
  python examples/timeseries_dataset_example.py --window 64 --stride 32
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from medvllm.data import get_dataset, MedicalDatasetConfig


def make_synthetic_csv(path: Path, rows: int = 256, features: int = 4) -> None:
    rng = np.random.default_rng(0)
    df = {
        "series_id": ["s1"] * rows,
        "t": np.arange(rows),
    }
    for i in range(features):
        df[f"f{i}"] = rng.standard_normal(size=rows).astype(np.float32)
    # target: next-step of f0 (shifted)
    y = np.roll(df["f0"], -1)
    y[-1] = y[-2]
    df["y"] = y
    import pandas as _pd

    _pd.DataFrame(df).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    csv_path: Path
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="ts_example_"))
        csv_path = tmpdir / "series.csv"
        make_synthetic_csv(csv_path)
        print(f"Generated synthetic CSV at: {csv_path}")

    cfg = MedicalDatasetConfig(
        name="timeseries_example",
        timeseries_path=str(csv_path),
        series_id_col="series_id",
        time_col="t",
        feature_cols=["f0", "f1", "f2", "f3"],
        target_col="y",
        window=args.window,
        stride=args.stride,
        normalization="zscore",
        drop_last_incomplete=False,
    )

    ds = get_dataset(cfg)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    batch = next(iter(dl))
    x, y, m = batch["input"], batch["target"], batch["mask"]
    print(f"input: {tuple(x.shape)}  target: {tuple(y.shape)}  mask: {tuple(m.shape)}")
    print(f"meta[0]: {batch['meta'][0]}")


if __name__ == "__main__":
    main()
