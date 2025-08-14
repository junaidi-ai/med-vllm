"""Medical model registration for the model registry.

This module provides functions to register medical models with the model registry,
including pre-configured models and their loaders with adapter support.
"""

from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from medvllm.engine.model_runner.registry import ModelType, registry
from medvllm.models.adapter import create_medical_adapter
from medvllm.models.medical_models import (
    BioBERTLoader,
    ClinicalBERTLoader,
    MedicalModelLoader,
)

# Define type variables for the model and tokenizer types
M = TypeVar("M", bound=PreTrainedModel)
T = TypeVar("T", bound=PreTrainedTokenizer)


class MedicalModelAdapterLoader(MedicalModelLoader[M, T]):
    """Loader that wraps a model with a medical adapter."""

    def __init__(self, model_id: str, model_type: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.model_type = model_type
        self.config = config
        self.MODEL_NAME = model_id
        self.MODEL_TYPE = model_type

        # Set the default model class from config or use AutoModel
        model_class = config.get("model_class")
        if model_class is not None:
            self.DEFAULT_MODEL_CLASS = model_class

        # Set the tokenizer class if specified in config
        tokenizer_class = config.get("tokenizer_class")
        if tokenizer_class is not None:
            self.TOKENIZER_CLASS = tokenizer_class

    @classmethod
    def load_model(
        cls,
        model_class: Optional[Type[M]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> Tuple[M, T]:
        """Load the model and wrap it with the appropriate adapter.

        Args:
            model_class: The model class to use (defaults to the one from config)
            config: Optional configuration overrides
            device: Device to load the model on
            **kwargs: Additional arguments to pass to the model loader

        Returns:
            The loaded model wrapped with the appropriate adapter
        """
        # Load the model and tokenizer using the parent class
        model, tokenizer = super().load_model(
            model_class=model_class, config=config or {}, device=device, **kwargs
        )

        # Create the appropriate adapter
        adapter = create_medical_adapter(
            model=model,
            model_type=cls.MODEL_TYPE,  # Use class attribute instead of instance
            config=config or {},
        )

        # Set up for inference by default
        adapter.setup_for_inference()

        # Return both the adapted model and tokenizer
        return adapter, tokenizer


# Registry for medical models
MEDICAL_MODEL_CONFIGS = {
    "dmis-lab/biobert-v1.1": {
        "model_class": BioBERTLoader.DEFAULT_MODEL_CLASS,
        "tokenizer_class": BioBERTLoader.TOKENIZER_CLASS,
        "config_class": None,  # Will be auto-detected
        "model_type": "biomedical",
        "description": "BioBERT v1.1 - Biomedical Language Model",
        "tags": ["biomedical", "biobert", "pretrained"],
    },
    "emilyalsentzer/Bio_ClinicalBERT": {
        "model_class": ClinicalBERTLoader.DEFAULT_MODEL_CLASS,
        "tokenizer_class": ClinicalBERTLoader.TOKENIZER_CLASS,
        "config_class": None,  # Will be auto-detected
        "model_type": "clinical",
        "description": "Clinical BERT - Pretrained on clinical notes",
        "tags": ["clinical", "bert", "pretrained"],
    },
}


def register_medical_models() -> None:
    """Register all pre-configured medical models with the registry.

    This function registers each medical model with its adapter, ensuring
    compatibility with the Nano vLLM architecture.
    """
    for model_id, config in MEDICAL_MODEL_CONFIGS.items():
        # Extract model metadata with proper typing
        model_type_str = str(config.get("model_type", "biomedical")).upper()
        description = str(config.get("description", ""))
        tags = config.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        params_raw: Dict[str, Any] | Any = config.get("parameters", {})
        params: Dict[str, Any] = params_raw if isinstance(params_raw, dict) else {}

        # Create a custom loader for this model
        model_type_enum: ModelType = ModelType[model_type_str]
        loader_class = type(
            f"{model_type_str.capitalize()}Loader",
            (MedicalModelAdapterLoader,),
            {"MODEL_TYPE": model_type_str, "MODEL_NAME": model_id},
        )

        # Get model and config classes with proper typing
        model_class = config.get("model_class")
        config_class = config.get("config_class")

        # Register the model with the registry
        registry.register(
            name=model_id,
            model_type=model_type_enum,
            model_class=model_class if isinstance(model_class, type) else None,
            config_class=config_class if isinstance(config_class, type) else None,
            description=description,
            tags=tags,
            loader=loader_class(model_id, model_type_str.lower(), config),
            **params,
        )


def register_custom_medical_model(
    name: str,
    model_name_or_path: str,
    model_type: ModelType,
    model_class: Optional[Type] = None,
    loader: Optional[Type[MedicalModelLoader]] = None,
    description: str = "",
    tags: Optional[list] = None,
    **parameters: Any,
) -> None:
    """Register a custom medical model with the registry.

    Args:
        name: Unique name for the model in the registry.
        model_name_or_path: Name or path of the pretrained model.
        model_type: Type of the model (BIOMEDICAL or CLINICAL).
        model_class: Optional custom model class to use.
        loader: Optional custom loader class for the model.
        description: Description of the model.
        tags: List of tags for the model.
        **parameters: Additional parameters to pass to the model.
    """
    # Include model_name_or_path in the parameters dictionary
    model_params = {"pretrained_model_name_or_path": model_name_or_path}
    if parameters:
        model_params.update(parameters)

    registry.register(
        name=name,
        model_type=model_type,
        model_class=model_class,
        description=description,
        tags=tags or [],
        loader=loader,
        parameters=model_params,
    )


# Register all pre-configured models when this module is imported
# Note: This is intentionally called only once at the module level
