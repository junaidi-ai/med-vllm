"""Configuration classes for medical datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List
import json


@dataclass
class MedicalDatasetConfig:
    """Configuration for a medical dataset.

    Attributes:
        name: Name of the dataset (e.g., "med_qa", "pubmed_qa")
        split: Dataset split to use ("train", "validation", "test")
        max_length: Maximum sequence length for tokenization (text)
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        cache_dir: Directory to cache processed data

        path: HF datasets path or identifier (for text datasets)
        text_column: Column name containing input text
        label_column: Column name containing labels (optional)

        data_dir: Root directory for imaging datasets
        image_format: One of {"dicom", "nifti", "png", "jpg"}
        pattern: Glob pattern to find files within data_dir
        annotation_path: Optional path to annotations mapping (e.g., JSON)
        is_3d: Whether data should be treated as 3D volumes
        normalization: One of {"none", "minmax", "zscore"}
        augment: Whether to apply simple augmentations
    """

    # Common
    name: str
    split: str = "train"
    max_length: int = 512
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    cache_dir: Optional[str] = None

    # Text datasets (HF datasets)
    path: Optional[str] = None
    text_column: Optional[str] = None
    label_column: Optional[str] = None

    # Imaging datasets (files on disk)
    data_dir: Optional[str] = None
    image_format: Optional[str] = None
    pattern: Optional[str] = None
    annotation_path: Optional[str] = None
    is_3d: Optional[bool] = None
    normalization: Optional[str] = None
    augment: bool = False
    # Optional: Triton kernel prototyping flags
    # Optional: Triton kernel prototyping flags
    enable_triton_ops: bool = False
    triton_op: Optional[str] = None  # e.g., 'median2d', 'boxblur2d'
    triton_window: int = 3

    # Large imaging handling (tiling / sliding window)
    # 2D tiling (for slices or 2D images)
    enable_tiling_2d: bool = False
    tile_size_2d: Optional[tuple[int, int]] = None  # (H, W)
    tile_stride_2d: Optional[tuple[int, int]] = None  # (H, W)
    # 3D tiling (for volumes)
    enable_tiling_3d: bool = False
    tile_size_3d: Optional[tuple[int, int, int]] = None  # (D, H, W)
    tile_stride_3d: Optional[tuple[int, int, int]] = None  # (D, H, W)

    # Time-series datasets (CSV files or directories of CSV)
    timeseries_path: Optional[str] = None  # single CSV file
    timeseries_dir: Optional[str] = None  # directory of CSV files
    file_pattern: Optional[str] = None  # glob within timeseries_dir, e.g., "*.csv"
    index_col: Optional[str] = None  # optional index column name
    time_col: Optional[str] = None  # timestamp column (optional)
    series_id_col: Optional[str] = None  # series/group id column (optional)
    feature_cols: Optional[List[str]] = None  # feature column names; if None, infer numeric
    target_col: Optional[str] = None  # optional target column name
    window: int = 128  # sliding window length
    stride: int = 64  # sliding stride
    horizon: int = 0  # prediction horizon (steps ahead)
    padding_value: float = 0.0  # padding value for short sequences
    drop_last_incomplete: bool = True  # drop windows shorter than "window" if True

    # Allow for additional dataset-specific parameters
    def __post_init__(self) -> None:
        # Normalize optional string fields
        if isinstance(self.cache_dir, str) and self.cache_dir.lower() == "none":
            self.cache_dir = None
        if isinstance(self.annotation_path, str) and self.annotation_path.lower() == "none":
            self.annotation_path = None
        if isinstance(self.data_dir, str) and self.data_dir.lower() == "none":
            self.data_dir = None
        if isinstance(self.timeseries_path, str) and self.timeseries_path.lower() == "none":
            self.timeseries_path = None
        if isinstance(self.timeseries_dir, str) and self.timeseries_dir.lower() == "none":
            self.timeseries_dir = None
        # Normalize imaging options
        if isinstance(self.normalization, str):
            self.normalization = self.normalization.lower()
        if isinstance(self.image_format, str):
            self.image_format = self.image_format.lower()
        if isinstance(self.triton_op, str):
            self.triton_op = self.triton_op.lower().strip()
        # Clamp window
        try:
            self.triton_window = int(self.triton_window)
        except Exception:
            self.triton_window = 3
        if self.triton_window < 1:
            self.triton_window = 1
        # Normalize tiling tuples if provided as lists
        # Keep as-is if None
        if self.tile_size_2d is not None:
            self.tile_size_2d = tuple(int(x) for x in self.tile_size_2d)  # type: ignore
        if self.tile_stride_2d is not None:
            self.tile_stride_2d = tuple(int(x) for x in self.tile_stride_2d)  # type: ignore
        if self.tile_size_3d is not None:
            self.tile_size_3d = tuple(int(x) for x in self.tile_size_3d)  # type: ignore
        if self.tile_stride_3d is not None:
            self.tile_stride_3d = tuple(int(x) for x in self.tile_stride_3d)  # type: ignore

    @classmethod
    def from_json_file(cls, config_path: str) -> "MedicalDatasetConfig":
        """Create a configuration from a JSON file."""
        path_obj = Path(config_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with path_obj.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return cls(**data)


def get_dataset_config(name: str, **kwargs) -> MedicalDatasetConfig:
    """Factory function to get a dataset configuration.

    Args:
        name: Name of the dataset
        **kwargs: Additional configuration parameters

    Returns:
        Configured MedicalDatasetConfig instance
    """
    return MedicalDatasetConfig(name=name, **kwargs)
