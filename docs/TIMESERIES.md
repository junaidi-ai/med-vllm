# Time-series Datasets and Benchmarking

This document explains how to use the new time-series dataset utilities, run the micro-benchmark, and interpret profiler outputs.

## Overview

- Dataset: `TimeSeriesDataset` in `medvllm/data/timeseries_datasets.py`
- Routing: `medvllm.data.get_dataset()` returns a `TimeSeriesDataset` when `MedicalDatasetConfig` includes `timeseries_path` or `timeseries_dir`.
- Formats: CSV (single file) or a directory of CSVs, optionally grouped by a `series_id` column.
- Sliding windows: controlled by `window` and `stride`, with horizon prediction via `horizon`.
- Normalization: per-series `none|minmax|zscore` using the shared `normalization` field.

## Configuration Fields (MedicalDatasetConfig)

Key fields for time-series use:
- `timeseries_path`: path to a single CSV file.
- `timeseries_dir`: directory with CSV files.
- `file_pattern`: glob pattern within `timeseries_dir` (default `*.csv`).
- `index_col`: optional DataFrame index column.
- `time_col`: optional timestamp column.
- `series_id_col`: optional column to group rows into separate series per file.
- `feature_cols`: list of feature column names. If omitted, numeric columns are auto-inferred (excluding `target`, `series_id`, `time`).
- `target_col`: optional target column name.
- `window`: sliding window length (default 128).
- `stride`: sliding stride (default 64).
- `horizon`: prediction horizon; target is taken at `end-1 + horizon` (default 0).
- `padding_value`: used when `drop_last_incomplete=False` and a window exceeds series length.
- `drop_last_incomplete`: if True, drop short windows; otherwise pad to `window`.

## Returned Item Structure

Each dataset item is a dict:
- `input`: FloatTensor `[window, F]`
- `target`: FloatTensor scalar (can be extended for multi-step)
- `mask`: FloatTensor `[window]` where 1=valid, 0=padded
- `meta`: `{series_id, idx_in_series, file_path}`

## Minimal Example

```
python examples/timeseries_dataset_example.py --window 64 --stride 32
```

This generates a synthetic CSV if `--csv` is not provided, loads the dataset via `get_dataset()`, and prints batch shapes/metadata.

## Benchmark

Run the micro-benchmark with unified profiler:

```
# Synthetic CSV auto-generated
python benchmarks/benchmark_timeseries.py --model gru --device cpu --iters 50

# Your CSV
python benchmarks/benchmark_timeseries.py \
  --csv path/to/series.csv \
  --features f0 f1 f2 \
  --model conv1d --device cuda --iters 100 --emit-trace --trace-dir ./profiles
```

Outputs JSON with throughput, latency, and profiler results. When `--emit-trace` is used on supported setups, profiler traces are saved under `--trace-dir`.

## Dependencies

- Requires `pandas` for CSV ingestion. If not installed, a clear ImportError will be raised when constructing `TimeSeriesDataset`.

## Notes and Extensions

- To support multi-step targets, extend the dataset to slice target sequences instead of scalars.
- For variable-length batching, a custom `collate_fn` can stack variable windows with additional padding per batch.
- Time encoding (positional or calendar features) can be added by transforming `time_col` into feature columns.
