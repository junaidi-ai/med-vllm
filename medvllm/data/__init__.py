"""Data loading and preprocessing for medical applications."""

from .medical_datasets import MedicalDataset, get_medical_dataset
from .imaging_datasets import ImagingDataset, get_imaging_dataset
from .tokenization.medical_tokenizer import MedicalTokenizer
from .config import MedicalDatasetConfig


def get_dataset(config: MedicalDatasetConfig | dict):
    """Return a dataset instance based on provided config.

    If `data_dir` or `image_format` is provided, returns `ImagingDataset`.
    Otherwise, returns `MedicalDataset` (HuggingFace text dataset).
    """
    if isinstance(config, dict):
        cfg = MedicalDatasetConfig(**config)
    else:
        cfg = config

    if cfg.data_dir or cfg.image_format:
        return ImagingDataset(cfg)
    return MedicalDataset(cfg)


__all__ = [
    "MedicalDataset",
    "get_medical_dataset",
    "ImagingDataset",
    "get_imaging_dataset",
    "get_dataset",
    "MedicalTokenizer",
]
