"""Utilities for loading and processing medical datasets."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def load_medical_dataset(
    dataset_name: str, split: str = "train", max_samples: Optional[int] = None, **kwargs
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """Load a medical dataset.

    This is a minimal implementation to satisfy test imports. In a real implementation,
    this would load the actual dataset from disk or a remote source.

    Args:
        dataset_name: Name of the dataset to load (e.g., "mimic-iii", "i2b2").
        split: Which split to load ("train", "val", "test").
        max_samples: Maximum number of samples to load (for testing).
        **kwargs: Additional dataset-specific arguments.

    Returns:
        A tuple of (features, labels) where both are dictionaries containing the dataset.
        Returns (features, None) if labels are not available.
    """
    # Placeholder implementation that returns empty data
    num_samples = max_samples or 10  # Default to 10 samples if max_samples is None
    seq_length = 128  # Default sequence length

    # Generate random data for testing
    input_ids = np.random.randint(0, 1000, size=(num_samples, seq_length))
    attention_mask = np.ones_like(input_ids)

    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # For testing, generate random labels if this is a training/validation split
    if split in ["train", "val"]:
        labels = np.random.randint(0, 2, size=num_samples)  # Binary classification
        return features, {"labels": labels}

    return features, None


class MedicalDataset(Dataset):
    """A PyTorch Dataset for medical data.

    This is a minimal implementation to satisfy test imports.
    """

    def __init__(
        self,
        features: Dict[str, np.ndarray],
        labels: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Initialize the dataset.

        Args:
            features: Dictionary of feature arrays.
            labels: Optional dictionary of label arrays.
        """
        self.features = features
        self.labels = labels or {}
        self.length = len(next(iter(features.values()))) if features else 0

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Get a single sample by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing the sample's features and labels.
        """
        sample = {}

        # Add features
        for key, value in self.features.items():
            sample[key] = value[idx]

        # Add labels if available
        for key, value in self.labels.items():
            sample[key] = value[idx]

        return sample
