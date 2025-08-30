import os
from pathlib import Path

import numpy as np
import pytest

from medvllm.data import get_dataset, MedicalDatasetConfig
from medvllm.data.timeseries_datasets import TimeSeriesDataset


FIXTURE_CSV = Path(__file__).resolve().parents[2] / "fixtures" / "data" / "timeseries_small.csv"


@pytest.mark.parametrize("window,stride", [(8, 8), (4, 2)])
def test_timeseries_routing_and_shapes(window: int, stride: int):
    cfg = MedicalDatasetConfig(
        name="ts_unit",
        timeseries_path=str(FIXTURE_CSV),
        series_id_col="series_id",
        time_col="t",
        feature_cols=["f0", "f1"],
        target_col="y",
        window=window,
        stride=stride,
        normalization="none",
        drop_last_incomplete=True,
        batch_size=2,
        shuffle=False,
    )
    ds = get_dataset(cfg)
    assert isinstance(ds, TimeSeriesDataset)
    item = ds[0]
    x, y, m = item["input"], item["target"], item["mask"]
    assert isinstance(x, np.ndarray) and x.shape == (window, 2)
    assert isinstance(m, np.ndarray) and m.shape == (window,)
    # y is numpy scalar or float
    assert np.isscalar(y) or (isinstance(y, np.ndarray) and y.shape == ())


def test_padding_when_window_exceeds_length():
    # CSV has 8 rows for single series
    cfg = MedicalDatasetConfig(
        name="ts_pad",
        timeseries_path=str(FIXTURE_CSV),
        series_id_col="series_id",
        time_col="t",
        feature_cols=["f0", "f1"],
        target_col="y",
        window=10,
        stride=10,
        normalization="none",
        drop_last_incomplete=False,
        padding_value=-1.0,
        shuffle=False,
    )
    ds = get_dataset(cfg)
    item = ds[0]
    x, y, m = item["input"], item["target"], item["mask"]
    # Expect mask with 8 valid then 2 padded
    assert np.allclose(m[:8], np.ones(8, dtype=np.float32))
    assert np.allclose(m[8:], np.zeros(2, dtype=np.float32))
    # Padded rows equal padding_value
    assert np.allclose(x[8:], np.full((2, 2), -1.0, dtype=np.float32))


def test_zscore_normalization_full_window_stats():
    # Use full-window to check per-series zscore stats ~ mean 0, std 1
    cfg = MedicalDatasetConfig(
        name="ts_norm",
        timeseries_path=str(FIXTURE_CSV),
        series_id_col="series_id",
        time_col="t",
        feature_cols=["f0", "f1"],
        target_col="y",
        window=8,
        stride=8,
        normalization="zscore",
        drop_last_incomplete=True,
        shuffle=False,
    )
    ds = get_dataset(cfg)
    item = ds[0]
    x = item["input"]
    # Column-wise mean ~ 0, std ~ 1 (numerical tolerance)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    assert np.allclose(mu, np.zeros_like(mu), atol=1e-6)
    assert np.allclose(sd, np.ones_like(sd), atol=1e-6)
