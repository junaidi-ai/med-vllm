"""Medical model registration for the model registry.

This module provides functions to register medical models with the model registry,
including pre-configured models and their loaders with adapter support.
"""

from typing import Any, Dict, Optional, Type, Tuple, Union

import torch

from ..engine.model_runner.registry import ModelType, registry
from .medical_models import BioBERTLoader, ClinicalBERTLoader, MedicalModelLoader
from .adapter import create_medical_adapter

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
    """Register all pre-configured medical models with the registry.
    
    This function registers each medical model with its adapter, ensuring
    compatibility with the Nano vLLM architecture.
    """
    for model_id, config in MEDICAL_MODEL_CONFIGS.items():
        # Create a wrapper function that will load the model and wrap it with the appropriate adapter
        def create_adapter_wrapper(model_id: str = model_id, config: Dict = config) -> callable:
            def wrapper(*args, **kwargs) -> torch.nn.Module:
                # Load the base model using the original loader
                loader = config["loader"]
                model = loader.load_model(*args, **kwargs)
                
                # Create the appropriate adapter
                model_type = "biobert" if "biobert" in model_id.lower() else "clinicalbert"
                adapter = create_medical_adapter(
                    model=model,
                    model_type=model_type,
                    config=config
                )
                
                # Set up for inference by default
                adapter.setup_for_inference()
                return adapter
            
            wrapper.__name__ = f"{model_id}_adapter_wrapper"
            return wrapper
        
        # Extract and validate model info with proper types
        name = str(config.get("name", ""))
        model_type = config.get("model_type", ModelType.GENERIC)
        description = str(config.get("description", ""))
        tags = config.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(tag) for tag in tags if tag is not None]
        
        # Get additional parameters, ensuring they're valid
        params = config.get("parameters", {})
        
        # Register the model with the adapter wrapper
        registry.register(
            name=model_id,
            model_type=model_type,
            model_class=config.get("model_class"),
            config_class=config.get("config_class"),
            description=description,
            tags=tags,
            loader=create_adapter_wrapper(),
            **params
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
