"""Configuration classes for medical datasets."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MedicalDatasetConfig:
    """Configuration for a medical dataset.

    Attributes:
        name: Name of the dataset (e.g., "med_qa", "pubmed_qa")
        split: Dataset split to use ("train", "validation", "test")
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        cache_dir: Directory to cache the dataset
        **kwargs: Additional dataset-specific configuration parameters
    """

    name: str
    split: str = "train"
    max_length: int = 512
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    cache_dir: Optional[str] = None

    # Allow for additional dataset-specific parameters
    def __post_init__(self):
        # Convert string "None" to actual None for optional fields
        if isinstance(self.cache_dir, str) and self.cache_dir.lower() == "none":
            self.cache_dir = None


def get_dataset_config(name: str, **kwargs) -> MedicalDatasetConfig:
    """Factory function to get a dataset configuration.

    Args:
        name: Name of the dataset
        **kwargs: Additional configuration parameters

    Returns:
        Configured MedicalDatasetConfig instance
    """
    return MedicalDatasetConfig(name=name, **kwargs)
