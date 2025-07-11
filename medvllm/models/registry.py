"""Medical model registration for the model registry.

This module provides functions to register medical models with the model registry,
including pre-configured models and their loaders.
"""

from typing import Any, Dict, Optional, Type

from ..engine.model_runner.registry import ModelType, registry
from .medical_models import BioBERTLoader, ClinicalBERTLoader, MedicalModelLoader

# Default model configurations
MEDICAL_MODEL_CONFIGS = {
    # BioBERT models
    "biobert-base": {
        "name": "dmis-lab/biobert-v1.1",
        "model_type": ModelType.BIOMEDICAL,
        "description": "BioBERT v1.1 - Pretrained biomedical language representation model",
        "tags": ["biomedical", "bert", "base"],
        "loader": BioBERTLoader,
        "parameters": {"num_labels": 2, "problem_type": "single_label_classification"},
    },
    # ClinicalBERT models
    "clinical-bert-base": {
        "name": "emilyalsentzer/Bio_ClinicalBERT",
        "model_type": ModelType.CLINICAL,
        "description": "Bio_ClinicalBERT - Pretrained clinical language representation model",
        "tags": ["clinical", "bert", "base"],
        "loader": ClinicalBERTLoader,
        "parameters": {"num_labels": 2, "problem_type": "single_label_classification"},
    },
    # Add more pre-configured medical models here
}


def register_medical_models() -> None:
    """Register all pre-configured medical models with the registry."""
    for model_name, model_info in MEDICAL_MODEL_CONFIGS.items():
        # Extract and validate model info with proper types
        name = str(model_info.get("name", ""))

        # Handle model_type
        model_type = model_info.get("model_type")
        if not isinstance(model_type, ModelType):
            try:
                model_type = ModelType(model_type)
            except (ValueError, TypeError):
                model_type = ModelType.GENERIC

        # Handle model_class and config_class
        model_class = model_info.get("model_class")
        if model_class is not None and not isinstance(model_class, type):
            model_class = None

        config_class = model_info.get("config_class")
        if config_class is not None and not isinstance(config_class, type):
            config_class = None

        description = str(model_info.get("description", ""))

        # Convert tags to list of strings
        tags = model_info.get("tags")
        if not isinstance(tags, list):
            tags = []
        tags = [str(tag) for tag in tags if tag is not None]

        # Get loader if it exists and is the correct type
        loader = model_info.get("loader")
        if loader is not None and not isinstance(loader, type):
            loader = None

        # Get additional parameters, ensuring they're valid
        params = model_info.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        # Ensure all parameter values are JSON-serializable
        params = {
            k: v
            for k, v in params.items()
            if isinstance(k, str) and isinstance(v, (str, int, float, bool, type(None)))
        }

        # Register the model with the registry
        try:
            registry.register(
                name=name,
                model_type=model_type,
                model_class=model_class,
                config_class=config_class,
                description=description,
                tags=tags,
                loader=loader,
                **params,
            )
        except Exception as e:
            print(f"Failed to register {model_name}: {str(e)}")
            continue


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
    registry.register(
        name=name,
        model_type=model_type,
        model_class=model_class,
        description=description,
        tags=tags or [],
        loader=loader,
        pretrained_model_name_or_path=model_name_or_path,
        **parameters,
    )


# Register all pre-configured models when this module is imported
register_medical_models()
