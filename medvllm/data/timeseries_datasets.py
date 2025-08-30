"""Time-series dataset utilities.

Provides a sliding-window TimeSeriesDataset that can read a single CSV file
or a directory of CSV files, optionally grouped by a series ID column.
Supports per-series normalization (none|minmax|zscore), horizon prediction,
and padding for short sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math

try:  # Optional dependency for convenient CSV reading
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

import numpy as np

from .config import MedicalDatasetConfig


@dataclass
class _SeriesMeta:
    series_id: str
    file_path: Path
    length: int
    feature_cols: List[str]
    target_col: Optional[str]


class TimeSeriesDataset:
    """Sliding-window time-series dataset.

    Returns items as dicts:
      - input: FloatTensor [window, F]
      - target: FloatTensor [] or [T] depending on horizon and target availability
      - mask: FloatTensor [window] (1 for valid, 0 for padded)
      - meta: dict with series_id, idx_in_series, file_path
    """

    def __init__(self, config: MedicalDatasetConfig) -> None:
        if pd is None:
            raise ImportError(
                "pandas is required for TimeSeriesDataset. Install with: pip install pandas"
            )
        self.cfg = config
        self._files: List[Path] = []
        if self.cfg.timeseries_path:
            self._files = [Path(self.cfg.timeseries_path)]
        elif self.cfg.timeseries_dir:
            pat = self.cfg.file_pattern or "*.csv"
            self._files = sorted(Path(self.cfg.timeseries_dir).glob(pat))
        else:
            raise ValueError("Provide either timeseries_path or timeseries_dir in config")
        if not self._files:
            raise FileNotFoundError("No CSV files found for TimeSeriesDataset")

        # Read files and build index of windows
        self._feature_cols: Optional[List[str]] = self.cfg.feature_cols
        self._target_col: Optional[str] = self.cfg.target_col

        self._series: List[Tuple[_SeriesMeta, np.ndarray, Optional[np.ndarray]]] = []
        for f in self._files:
            df = (
                pd.read_csv(f, index_col=self.cfg.index_col)
                if self.cfg.index_col
                else pd.read_csv(f)
            )
            # If series_id_col provided, split into groups; else single series per file
            if self.cfg.series_id_col and self.cfg.series_id_col in df.columns:
                for sid, g in df.groupby(self.cfg.series_id_col):
                    xcols = self._infer_feature_cols(g)
                    ycol = (
                        self._target_col
                        if (self._target_col and self._target_col in g.columns)
                        else None
                    )
                    X = g[xcols].to_numpy(dtype=np.float32, copy=False)
                    y = g[ycol].to_numpy(dtype=np.float32, copy=False) if ycol else None
                    X, y = self._maybe_normalize(X, y)
                    meta = _SeriesMeta(
                        series_id=str(sid),
                        file_path=f,
                        length=len(g),
                        feature_cols=xcols,
                        target_col=ycol,
                    )
                    self._series.append((meta, X, y))
            else:
                xcols = self._infer_feature_cols(df)
                ycol = (
                    self._target_col
                    if (self._target_col and self._target_col in df.columns)
                    else None
                )
                X = df[xcols].to_numpy(dtype=np.float32, copy=False)
                y = df[ycol].to_numpy(dtype=np.float32, copy=False) if ycol else None
                X, y = self._maybe_normalize(X, y)
                meta = _SeriesMeta(
                    series_id=f.stem,
                    file_path=f,
                    length=len(df),
                    feature_cols=xcols,
                    target_col=ycol,
                )
                self._series.append((meta, X, y))

        # Precompute window index mapping: (series_idx, start_idx)
        self._index: List[Tuple[int, int]] = []
        W = int(self.cfg.window)
        S = int(self.cfg.stride)
        drop_incomplete = bool(self.cfg.drop_last_incomplete)
        for si, (meta, X, _y) in enumerate(self._series):
            L = X.shape[0]
            if L == 0:
                continue
            if L < W:
                if drop_incomplete:
                    continue
                else:
                    # Still create a single padded window starting at 0
                    self._index.append((si, 0))
                    continue
            # Sliding windows
            pos = 0
            while pos + W <= L:
                self._index.append((si, pos))
                pos += S if S > 0 else W  # avoid infinite loops
            if not drop_incomplete and pos < L:
                self._index.append((si, L - W))

    def _infer_feature_cols(self, df: "pd.DataFrame") -> List[str]:
        if self._feature_cols:
            missing = [c for c in self._feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
            return list(self._feature_cols)
        # Infer numeric columns excluding target and series/time columns
        exclude = set(
            [c for c in [self.cfg.target_col, self.cfg.series_id_col, self.cfg.time_col] if c]
        )
        num_cols = [
            c for c, dt in df.dtypes.items() if np.issubdtype(dt, np.number) and c not in exclude
        ]
        if not num_cols:
            raise ValueError("Could not infer numeric feature columns from CSV")
        return num_cols

    def _maybe_normalize(
        self, X: np.ndarray, y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        norm = (self.cfg.normalization or "none").lower()
        if norm == "none":
            return X, y
        if norm == "minmax":
            x_min = X.min(axis=0, keepdims=True)
            x_max = X.max(axis=0, keepdims=True)
            denom = np.where((x_max - x_min) == 0.0, 1.0, (x_max - x_min))
            Xn = (X - x_min) / denom
            return Xn.astype(np.float32, copy=False), y
        if norm == "zscore":
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True)
            sd = np.where(sd == 0.0, 1.0, sd)
            Xn = (X - mu) / sd
            return Xn.astype(np.float32, copy=False), y
        return X, y

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        si, start = self._index[idx]
        meta, X, y = self._series[si]
        W = int(self.cfg.window)
        H = int(self.cfg.horizon)
        L = X.shape[0]

        end = start + W
        x_slice = X[start:end]
        valid = min(W, max(0, L - start))
        if x_slice.shape[0] < W:
            pad_len = W - x_slice.shape[0]
            pad = np.full((pad_len, X.shape[1]), self.cfg.padding_value, dtype=np.float32)
            x_slice = np.concatenate([x_slice, pad], axis=0)
        mask = np.zeros((W,), dtype=np.float32)
        mask[:valid] = 1.0

        if y is not None:
            y_idx = min(L - 1, end - 1 + H)
            target = np.asarray(y[y_idx], dtype=np.float32)
        else:
            target = np.asarray(0.0, dtype=np.float32)

        item = {
            "input": x_slice,  # numpy array [W, F]
            "target": target,  # numpy scalar/array
            "mask": mask,  # numpy array [W]
            "meta": {
                "series_id": meta.series_id,
                "idx_in_series": int(start),
                "file_path": str(meta.file_path),
            },
        }
        return item


def get_timeseries_dataset(config: MedicalDatasetConfig | Dict[str, Any]) -> TimeSeriesDataset:
    if isinstance(config, dict):
        cfg = MedicalDatasetConfig(**config)
    else:
        cfg = config
    return TimeSeriesDataset(cfg)
