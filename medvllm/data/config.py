"""Configuration classes for medical datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
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

    # Allow for additional dataset-specific parameters
    def __post_init__(self) -> None:
        # Normalize optional string fields
        if isinstance(self.cache_dir, str) and self.cache_dir.lower() == "none":
            self.cache_dir = None
        if isinstance(self.annotation_path, str) and self.annotation_path.lower() == "none":
            self.annotation_path = None
        if isinstance(self.data_dir, str) and self.data_dir.lower() == "none":
            self.data_dir = None

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
