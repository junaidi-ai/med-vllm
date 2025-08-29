"""Medical dataset implementations and utilities."""

from typing import Any, Dict, Optional, Union, TYPE_CHECKING

import datasets
import torch

# Optional transformers import: only required when using text datasets with tokenization
try:  # pragma: no cover - exercised via integration
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore

if TYPE_CHECKING:  # for type checkers only
    from transformers import PreTrainedTokenizerBase  # noqa: F401
else:
    PreTrainedTokenizerBase = Any  # runtime fallback to avoid hard dependency

# Import the config class
from medvllm.data.config import MedicalDatasetConfig


class MedicalDataset:
    """Base class for medical datasets."""

    def __init__(
        self,
        config: MedicalDatasetConfig,
        tokenizer: Optional[Union[PreTrainedTokenizerBase, str]] = None,
        tokenizer_name: str = "bert-base-uncased",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        if isinstance(tokenizer, str):
            if AutoTokenizer is None:
                raise ImportError(
                    "transformers is required to load tokenizer by name. Install with: pip install transformers"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **(tokenizer_kwargs or {}))
        elif tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if AutoTokenizer is None:
                raise ImportError(
                    "transformers is required for MedicalDataset tokenization. Install with: pip install transformers"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, **(tokenizer_kwargs or {})
            )
        self.dataset = self._load_dataset()

    @classmethod
    def from_config(
        cls, config_path: str, tokenizer: Optional[PreTrainedTokenizerBase] = None
    ) -> "MedicalDataset":
        """Create a dataset from a configuration file.

        Args:
            config_path: Path to the configuration file
            tokenizer: Optional tokenizer instance

        Returns:
            Configured MedicalDataset instance
        """
        config = MedicalDatasetConfig.from_json_file(config_path)
        return cls(config, tokenizer=tokenizer)

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> "MedicalDataset":
        """Create a dataset from a configuration dictionary.

        Args:
            config_dict: Configuration dictionary
            tokenizer: Optional tokenizer instance

        Returns:
            Configured MedicalDataset instance
        """
        config = MedicalDatasetConfig(**config_dict)
        return cls(config, tokenizer=tokenizer)

    def _load_dataset(self) -> datasets.Dataset:
        """Load the dataset using Hugging Face datasets."""
        try:
            dataset = datasets.load_dataset(
                self.config.path,
                split=self.config.split,
                cache_dir=getattr(self.config, "cache_dir", None),
            )
            return dataset
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {str(e)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        # Tokenize text
        text = item[self.config.text_column]
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {"input_ids": inputs["input_ids"].squeeze(0)}

        # Add labels if available
        if self.config.label_column and self.config.label_column in item:
            result["labels"] = torch.tensor(item[self.config.label_column])

        return result


def get_medical_dataset(name: str, **kwargs: Any) -> MedicalDataset:
    """Factory function to get a medical dataset.

    Args:
        name: Name of the dataset (e.g., "med_qa", "pubmed_qa")
        **kwargs: Additional arguments to pass to the dataset config

    Returns:
        Configured MedicalDataset instance
    """
    # Import here to avoid circular imports
    from medvllm.data.config import MedicalDatasetConfig

    # Define supported datasets
    DATASET_CONFIGS = {
        "med_qa": {
            "name": "med_qa",
            "path": "bigbio/med_qa",
            "text_column": "question",
            "label_column": "answer",
        },
        "pubmed_qa": {
            "name": "pubmed_qa",
            "path": "pubmed_qa",
            "text_column": "question",
            "label_column": "long_answer",
        },
        "mimic_iii": {
            "name": "mimic_iii",
            "path": "mimic_iii_discharge_summary",
            "text_column": "text",
        },
    }

    if name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {name}. " f"Available datasets: {list(DATASET_CONFIGS.keys())}"
        )

    # Create config from dictionary and update with any kwargs
    config_dict = DATASET_CONFIGS[name].copy()
    config_dict.update(kwargs)
    config = MedicalDatasetConfig(**config_dict)

    return MedicalDataset(config)
