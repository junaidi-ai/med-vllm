"""Medical model loaders for specialized healthcare NLP models.

This module provides specialized loaders for medical language models like BioBERT and ClinicalBERT,
handling model-specific configurations and tokenization requirements.
"""

from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

ModelT = TypeVar("ModelT", bound=PreTrainedModel)


class MedicalModelLoader:
    """Base class for medical model loaders."""

    MODEL_NAME: str
    MODEL_TYPE: str
    DEFAULT_MODEL_CLASS: Type[PreTrainedModel] = AutoModel
    TOKENIZER_CLASS: Type[PreTrainedTokenizer] = AutoTokenizer

    @classmethod
    def load_model(
        cls,
        model_class: Optional[Type[ModelT]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> Tuple[ModelT, PreTrainedTokenizer]:
        """Load the model and tokenizer with default medical-specific parameters.

        Args:
            model_class: Optional custom model class to use.
            config: Optional model configuration overrides.
            device: Device to load the model on (e.g., 'cuda', 'cpu').
            **kwargs: Additional arguments to pass to model and tokenizer.

        Returns:
            A tuple of (model, tokenizer)
        """
        # Load tokenizer first
        tokenizer = cls.load_tokenizer(**kwargs)

        # Load model
        model_class = model_class or cls.DEFAULT_MODEL_CLASS
        model = cls._load_model(model_class, config, device, **kwargs)

        return model, tokenizer

    @classmethod
    def _load_model(
        cls,
        model_class: Type[ModelT],
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> ModelT:
        """Internal method to load the model."""
        model_config = AutoConfig.from_pretrained(cls.MODEL_NAME, **kwargs)

        # Apply any config overrides
        if config:
            for key, value in config.items():
                setattr(model_config, key, value)

        model = model_class.from_pretrained(
            cls.MODEL_NAME,
            config=model_config,
            **{k: v for k, v in kwargs.items() if k not in model_config.to_diff_dict()},
        )

        if device:
            model = model.to(device)

        model.eval()
        return model

    @classmethod
    def load_tokenizer(cls, **kwargs: Any) -> PreTrainedTokenizer:
        """Load the tokenizer with default parameters.

        Args:
            **kwargs: Additional arguments to pass to the tokenizer.

        Returns:
            A pre-trained tokenizer instance.
        """
        return cls.TOKENIZER_CLASS.from_pretrained(cls.MODEL_NAME, **kwargs)


class BioBERTLoader(MedicalModelLoader):
    """Loader for BioBERT models."""

    MODEL_NAME = "dmis-lab/biobert-v1.1"
    MODEL_TYPE = "biomedical"
    DEFAULT_MODEL_CLASS = AutoModelForSequenceClassification

    @classmethod
    def load_model(
        cls,
        model_class: Optional[Type[ModelT]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> Tuple[ModelT, PreTrainedTokenizer]:
        """Load BioBERT model and tokenizer.

        Args:
            model_class: Optional custom model class to use.
            config: Optional model configuration overrides.
            device: Device to load the model on.
            **kwargs: Additional arguments to pass to model and tokenizer.

        Returns:
            A tuple of (model, tokenizer)
        """
        # Set default arguments for BioBERT
        kwargs.setdefault("num_labels", 2)  # Default binary classification
        return super().load_model(model_class, config, device, **kwargs)


class ClinicalBERTLoader(MedicalModelLoader):
    """Loader for ClinicalBERT models."""

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    MODEL_TYPE = "clinical"
    DEFAULT_MODEL_CLASS = AutoModelForSequenceClassification

    @classmethod
    def load_model(
        cls,
        model_class: Optional[Type[ModelT]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> Tuple[ModelT, PreTrainedTokenizer]:
        """Load ClinicalBERT model and tokenizer.

        Args:
            model_class: Optional custom model class to use.
            config: Optional model configuration overrides.
            device: Device to load the model on.
            **kwargs: Additional arguments to pass to model and tokenizer.

        Returns:
            A tuple of (model, tokenizer)
        """
        # Set default arguments for ClinicalBERT
        kwargs.setdefault("num_labels", 2)  # Default binary classification
        return super().load_model(model_class, config, device, **kwargs)
