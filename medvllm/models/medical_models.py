"""Medical model loaders for specialized healthcare NLP models.

This module provides specialized loaders for medical language models like BioBERT and ClinicalBERT,
handling model-specific configurations and tokenization requirements.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import torch
from torch import device

# Lazy imports for transformers
try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerBase,
        PreTrainedTokenizerFast,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Define dummy classes for type checking
    class PretrainedConfig:
        pass
    
    class PreTrainedModel:
        pass
    
    class PreTrainedTokenizer:
        pass
    
    class PreTrainedTokenizerBase:
        pass
    
    class PreTrainedTokenizerFast:
        pass
    
    # Define dummy variables for the Auto* classes
    AutoConfig = None
    AutoModel = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

# Import Protocol from typing_extensions if available
try:
    from typing_extensions import Protocol, runtime_checkable
except ImportError:
    from typing import Protocol, runtime_checkable  # type: ignore

# Define type variables for model and tokenizer types with proper bounds
ModelT = TypeVar("ModelT", bound=PreTrainedModel)

# Define a type alias for tokenizers
MedicalTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# Define a type variable for tokenizers
TokenizerT = TypeVar("TokenizerT", bound=MedicalTokenizer)

# Define a concrete model type that we'll use for our loaders
MedicalModel = TypeVar("MedicalModel", bound=PreTrainedModel)

# Define a type alias for the model class type
ModelClassType = Type[PreTrainedModel]


# Define a protocol for model classes that can be used with our loaders
class SupportsFromPretrained(Protocol):
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ) -> Any: ...


class MedicalModelLoader(Generic[ModelT, TokenizerT]):
    """Base class for medical model loaders.

    This class provides a generic interface for loading medical language models
    with proper type hints and configuration handling.

    Generic Types:
        ModelT: The type of model this loader handles, must be a subclass of PreTrainedModel
        TokenizerT: The type of tokenizer this loader uses, must be a subclass of MedicalTokenizer
    """

    MODEL_NAME: str
    MODEL_TYPE: str
    DEFAULT_MODEL_CLASS: Type[ModelT] = cast(Type[ModelT], AutoModel)
    TOKENIZER_CLASS: Type[TokenizerT] = cast(Type[TokenizerT], AutoTokenizer)

    @classmethod
    def load_model(
        cls,
        model_class: Optional[Type[ModelT]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> Tuple[ModelT, TokenizerT]:
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

        return model, tokenizer  # type: ignore[return-value]

    @classmethod
    def _load_model(
        cls,
        model_class: Type[ModelT],
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> ModelT:
        # Ensure model_class is properly typed and fall back to default if None
        model_class = model_class or cls.DEFAULT_MODEL_CLASS
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
    def load_tokenizer(cls, **kwargs: Any) -> TokenizerT:
        """Load the tokenizer with the specified parameters.

        Args:
            **kwargs: Additional keyword arguments to pass to the tokenizer's from_pretrained method

        Returns:
            Loaded tokenizer
        """
        tokenizer = cls.TOKENIZER_CLASS.from_pretrained(cls.MODEL_NAME, **kwargs)
        return cast(TokenizerT, tokenizer)


class BioBERTLoader(MedicalModelLoader[PreTrainedModel, MedicalTokenizer]):
    """Loader for BioBERT models with medical-specific optimizations.

    This loader handles the initialization and configuration of BioBERT models
    for biomedical NLP tasks, with support for the adapter interface.
    """

    MODEL_NAME = "dmis-lab/biobert-v1.1"
    MODEL_TYPE = "biomedical"
    DEFAULT_MODEL_CLASS: Type[PreTrainedModel] = AutoModelForSequenceClassification  # type: ignore[assignment]
    TOKENIZER_CLASS: Type[MedicalTokenizer] = AutoTokenizer  # type: ignore[assignment]

    @classmethod
    def load_model(
        cls,
        model_class: Optional[Type[PreTrainedModel]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> Tuple[PreTrainedModel, MedicalTokenizer]:
        """Load a BioBERT model and tokenizer with medical-specific optimizations.

        Args:
            model_class: The model class to use (default: AutoModelForSequenceClassification)
            config: Optional configuration dictionary
            device: Device to load the model on (e.g., 'cuda', 'cpu')
            **kwargs: Additional arguments to pass to the model and tokenizer

        Returns:
            A tuple of (model, tokenizer)
        """
        # Use AutoModelForSequenceClassification as the default model class if not specified
        model_class_to_use: Type[PreTrainedModel] = (
            AutoModelForSequenceClassification  # type: ignore[assignment]
            if model_class is None
            else model_class
        )

        # Load the model and tokenizer using the parent class method
        model, tokenizer = super().load_model(
            model_class=model_class_to_use, config=config, device=device, **kwargs
        )

        # Add any special tokens needed for biomedical tasks
        cls._add_special_tokens(tokenizer)

        # Resize token embeddings if needed
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))  # type: ignore[arg-type]

        return model, tokenizer

    @classmethod
    def _add_special_tokens(
        cls, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    ) -> None:
        """Add any special tokens needed for biomedical tasks.

        Args:
            tokenizer: The tokenizer to add special tokens to
        """
        # Add biomedical special tokens
        new_tokens = [
            "<DIAGNOSIS>",
            "<TREATMENT>",
            "<MEDICATION>",
            "<DOSAGE>",
            "<FREQUENCY>",
            "<DURATION>",
            "<ROUTE>",
            "<STRENGTH>",
            "<ADVERSE_EVENT>",
            "<LAB_TEST>",
            "<LAB_VALUE>",
            "<LAB_UNIT>",
            "<BODY_PART>",
            "<PROCEDURE>",
            "<CONDITION>",
            "<SYMPTOM>",
        ]

        # For PreTrainedTokenizer
        if isinstance(tokenizer, PreTrainedTokenizer):
            vocab = tokenizer.get_vocab()
            new_tokens = [token for token in new_tokens if token not in vocab]
        # For PreTrainedTokenizerFast
        elif hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()  # type: ignore
            new_tokens = [token for token in new_tokens if token not in vocab]

        if new_tokens and hasattr(tokenizer, "add_tokens"):
            tokenizer.add_tokens(new_tokens)  # type: ignore


class ClinicalBERTLoader(MedicalModelLoader[PreTrainedModel, MedicalTokenizer]):
    """Loader for ClinicalBERT models with clinical domain optimizations.

    This loader handles the initialization and configuration of ClinicalBERT models
    for clinical NLP tasks, with support for the adapter interface.
    """

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    MODEL_TYPE = "clinical"
    DEFAULT_MODEL_CLASS: Type[PreTrainedModel] = AutoModelForSequenceClassification  # type: ignore[assignment]
    TOKENIZER_CLASS: Type[MedicalTokenizer] = AutoTokenizer  # type: ignore[assignment]

    @classmethod
    def load_model(
        cls,
        model_class: Optional[Type[PreTrainedModel]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> Tuple[PreTrainedModel, MedicalTokenizer]:
        """Load a ClinicalBERT model and tokenizer with clinical-specific optimizations.

        Args:
            model_class: The model class to use (defaults to AutoModelForSequenceClassification)
            config: Optional model configuration
            device: Device to load the model on
            **kwargs: Additional arguments to pass to the model and tokenizer

        Returns:
            A tuple of (model, tokenizer)
        """
        # Load tokenizer first
        tokenizer = cls.load_tokenizer(**kwargs)

        # Load model with the appropriate class
        model_class = model_class or cls.DEFAULT_MODEL_CLASS
        model = cls._load_model(model_class, config, device, **kwargs)

        # Resize token embeddings if needed
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))  # type: ignore[arg-type]

        return model, tokenizer  # type: ignore[return-value]
