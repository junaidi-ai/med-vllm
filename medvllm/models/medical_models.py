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
    """Loader for BioBERT models with medical-specific optimizations.
    
    This loader handles the initialization and configuration of BioBERT models
    for biomedical NLP tasks, with support for the adapter interface.
    """
    
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
        """Load BioBERT model and tokenizer with medical-specific defaults.
        
        Args:
            model_class: Optional custom model class to use.
            config: Optional model configuration overrides.
            device: Device to load the model on.
            **kwargs: Additional arguments to pass to model and tokenizer.
                - tokenizer_kwargs: Additional arguments for the tokenizer
                - model_kwargs: Additional arguments for the model
                
        Returns:
            A tuple of (model, tokenizer)
        """
        # Set default tokenizer parameters for biomedical text
        tokenizer_kwargs = {
            "do_lower_case": False,  # BioBERT uses cased tokenization
            "model_max_length": 512,  # Maximum sequence length
            "use_fast": True,        # Use fast tokenizer
            "padding_side": "right", # For compatibility with generation
        }
        tokenizer_kwargs.update(kwargs.pop("tokenizer_kwargs", {}))
        
        # Set default model parameters
        model_kwargs = {
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,  # Use dict outputs for better compatibility
        }
        model_kwargs.update(kwargs.pop("model_kwargs", {}))
        
        # Load tokenizer with biomedical-specific settings
        tokenizer = cls.load_tokenizer(**tokenizer_kwargs)
        
        # Add special tokens for biomedical tasks if needed
        cls._add_special_tokens(tokenizer)
        
        # Load model with medical-specific defaults
        model_class = model_class or cls.DEFAULT_MODEL_CLASS
        model = cls._load_model(
            model_class=model_class,
            config=config,
            device=device,
            **model_kwargs
        )
        
        # Ensure model's embeddings match tokenizer vocab size
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    
    @classmethod
    def _add_special_tokens(cls, tokenizer: PreTrainedTokenizer) -> None:
        """Add any special tokens needed for biomedical tasks.
        
        Args:
            tokenizer: The tokenizer to add special tokens to
        """
        # Add common biomedical special tokens
        special_tokens = [
            "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]",
            "<e1>", "</e1>", "<e2>", "</e2>"  # For entity markers
        ]
        
        # Only add tokens that aren't already in the vocabulary
        new_tokens = [tok for tok in special_tokens 
                     if tok not in tokenizer.get_vocab()]
        
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
    
    @classmethod
    def _load_model(
        cls,
        model_class: Type[ModelT],
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> ModelT:
        """Internal method to load the model with medical-specific settings.
        
        Args:
            model_class: The model class to instantiate
            config: Optional configuration overrides
            device: Device to load the model on
            **kwargs: Additional model arguments
            
        Returns:
            The loaded model
        """
        # Load model config with medical defaults
        model_config = AutoConfig.from_pretrained(cls.MODEL_NAME, **kwargs)
        
        # Apply any config overrides
        if config:
            for key, value in config.items():
                setattr(model_config, key, value)
        
        # Initialize model with config
        model = model_class.from_pretrained(
            cls.MODEL_NAME,
            config=model_config,
            **{k: v for k, v in kwargs.items() if k not in model_config.to_diff_dict()},
        )
        
        # Move to device if specified
        if device:
            model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        return model


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
