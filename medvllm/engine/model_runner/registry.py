"""
Model Registry for managing and loading different model types.

This module provides a thread-safe registry for managing various model types,
including medical models like BioBERT and ClinicalBERT, with support for
model loading, caching, and metadata management.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union, overload

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig

from .types import PretrainedConfigT

# Type variable for model types
M = TypeVar('M', bound=PreTrainedModel)


class ModelType(Enum):
    """Supported model types in the registry."""
    GENERIC = auto()
    BIOMEDICAL = auto()
    CLINICAL = auto()


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    
    name: str
    """The unique name of the model."""
    
    model_type: ModelType = ModelType.GENERIC
    """The type of the model (e.g., BIOMEDICAL, CLINICAL)."""
    
    model_class: Optional[Type[PreTrainedModel]] = None
    """The model class to use for instantiation."""
    
    config_class: Optional[Type[PretrainedConfig]] = None
    """The config class to use for the model."""
    
    description: str = ""
    """Optional description of the model."""
    
    tags: list[str] = field(default_factory=list)
    """List of tags for categorizing the model."""
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    """Additional model parameters and configuration."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary."""
        return {
            'name': self.name,
            'model_type': self.model_type.name,
            'model_class': self.model_class.__name__ if self.model_class else None,
            'config_class': self.config_class.__name__ if self.config_class else None,
            'description': self.description,
            'tags': self.tags,
            'parameters': self.parameters
        }


class ModelRegistry(Generic[M]):
    """
    A thread-safe registry for managing model loading and instantiation.
    
    This class implements the singleton pattern to ensure there's only one registry instance.
    It provides methods for registering, retrieving, and managing models with their metadata.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._models: Dict[str, ModelMetadata] = {}
                cls._instance._model_cache: Dict[str, M] = {}
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._models = {}
            self._model_cache = {}
            self._initialized = True
            self._register_default_models()
    
    def _register_default_models(self) -> None:
        """Register default models with the registry."""
        try:
            from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
            
            # Register BioBERT models
            biobert_models = [
                {
                    "name": "dmis-lab/biobert-base-cased-v1.2",
                    "type": ModelType.BIOMEDICAL,
                    "description": "BioBERT: a pre-trained biomedical language representation model",
                    "tags": ["biomedical", "bert", "pre-trained"],
                },
                {
                    "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "type": ModelType.BIOMEDICAL,
                    "description": "PubMedBERT: A pre-trained language model for biomedical text mining",
                    "tags": ["biomedical", "bert", "pubmed", "pre-trained"],
                },
                {
                    "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                    "type": ModelType.BIOMEDICAL,
                    "description": "PubMedBERT trained on PubMed abstracts",
                    "tags": ["biomedical", "bert", "pubmed", "abstracts"],
                },
            ]
            
            # Register clinical models
            clinical_models = [
                {
                    "name": "emilyalsentzer/Bio_ClinicalBERT",
                    "type": ModelType.CLINICAL,
                    "description": "ClinicalBERT: Pretrained Clinical Language Model",
                    "tags": ["clinical", "bert", "pre-trained"],
                },
                {
                    "name": "bvanaken/clinical-assertion-negation-bert",
                    "type": ModelType.CLINICAL,
                    "model_class": AutoModelForSequenceClassification,
                    "description": "Clinical assertion and negation detection model",
                    "tags": ["clinical", "bert", "assertion", "negation"],
                },
                {
                    "name": "bvanaken/clinical-assertion-negation-bioclinicalbert",
                    "type": ModelType.CLINICAL,
                    "model_class": AutoModelForSequenceClassification,
                    "description": "BioClinicalBERT fine-tuned for assertion and negation",
                    "tags": ["clinical", "bert", "assertion", "negation", "bioclinicalbert"],
                },
            ]
            
            # Register all models
            for model_info in biobert_models + clinical_models:
                self.register(
                    name=model_info["name"],
                    model_type=model_info["type"],
                    model_class=model_info.get("model_class", AutoModelForCausalLM),
                    description=model_info["description"],
                    tags=model_info["tags"],
                    pretrained_model_name_or_path=model_info["name"]
                )
            
            # Register some general medical models
            medical_models = [
                {
                    "name": "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
                    "type": ModelType.BIOMEDICAL,
                    "description": "Large PubMedBERT model trained on PubMed abstracts",
                    "tags": ["biomedical", "bert", "pubmed", "large"],
                },
                {
                    "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "type": ModelType.BIOMEDICAL,
                    "description": "PubMedBERT trained on full-text articles",
                    "tags": ["biomedical", "bert", "pubmed", "fulltext"],
                },
            ]
            
            for model_info in medical_models:
                self.register(
                    name=model_info["name"],
                    model_type=model_info["type"],
                    model_class=AutoModelForCausalLM,
                    description=model_info["description"],
                    tags=model_info["tags"],
                    pretrained_model_name_or_path=model_info["name"]
                )
            
        except ImportError:
            pass  # Skip default registration if transformers is not available
    
    def register(
        self,
        name: str,
        model_type: ModelType = ModelType.GENERIC,
        model_class: Optional[Type[M]] = None,
        config_class: Optional[Type[PretrainedConfig]] = None,
        description: str = "",
        tags: Optional[list[str]] = None,
        **parameters: Any
    ) -> None:
        """Register a new model with the registry.
        
        Args:
            name: Unique name for the model.
            model_type: Type of the model (e.g., BIOMEDICAL, CLINICAL).
            model_class: The model class to use for instantiation.
            config_class: The config class to use for the model.
            description: Optional description of the model.
            tags: List of tags for categorizing the model.
            **parameters: Additional parameters to pass to the model during loading.
            
        Note:
            If the model is already registered, this method will skip the registration
            and return without raising an error.
        """
        with self._lock:
            if name in self._models:
                # Skip if already registered
                return
                
            self._models[name] = ModelMetadata(
                name=name,
                model_type=model_type,
                model_class=model_class,
                config_class=config_class,
                description=description,
                tags=tags or [],
                parameters=parameters or {}
            )
    
    def unregister(self, name: str) -> None:
        """Unregister a model from the registry.
        
        Args:
            name: Name of the model to unregister.
            
        Raises:
            KeyError: If the model is not found in the registry.
        """
        with self._lock:
            if name not in self._models:
                raise KeyError(f"Model '{name}' not found in registry")
                
            # Remove from cache if loaded
            if name in self._model_cache:
                del self._model_cache[name]
                
            # Remove from registry
            del self._models[name]
    
    def get_metadata(self, name: str) -> ModelMetadata:
        """Get metadata for a registered model.
        
        Args:
            name: Name of the model.
            
        Returns:
            The model's metadata.
            
        Raises:
            KeyError: If the model is not found in the registry.
        """
        with self._lock:
            if name not in self._models:
                raise KeyError(f"Model '{name}' not found in registry")
            return self._models[name]
    
    def list_models(self, model_type: Optional[ModelType] = None) -> Dict[str, ModelMetadata]:
        """List all registered models, optionally filtered by type.
        
        Args:
            model_type: If provided, only return models of this type.
            
        Returns:
            Dictionary mapping model names to their metadata.
        """
        with self._lock:
            if model_type is None:
                return self._models.copy()
            return {k: v for k, v in self._models.items() if v.model_type == model_type}
    
    def is_registered(self, name: str) -> bool:
        """Check if a model is registered.
        
        Args:
            name: Name of the model to check.
            
        Returns:
            True if the model is registered, False otherwise.
        """
        with self._lock:
            return name in self._models
    
    @overload
    def load_model(
        self,
        name: str,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any
    ) -> M:
        ...
        
    @overload
    def load_model(
        self,
        name: str,
        config: Optional[PretrainedConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any
    ) -> M:
        ...
    
    def load_model(
        self,
        name: str,
        config: Optional[PretrainedConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any
    ) -> M:
        """Load a model by name.
        
        Args:
            name: Name of the model to load.
            config: Optional model configuration. If not provided, will be loaded from the registry.
            device: Device to load the model onto. If None, will use the default device.
            **kwargs: Additional arguments to pass to the model's from_pretrained method.
            
        Returns:
            The loaded model instance.
            
        Raises:
            KeyError: If the model is not found in the registry.
            RuntimeError: If there's an error loading the model.
        """
        # Get model metadata
        try:
            metadata = self.get_metadata(name)
        except KeyError as e:
            # If model is not registered but is a valid model identifier, try to load it directly
            try:
                model = AutoModel.from_pretrained(name, **kwargs)
                if device is not None:
                    model = model.to(device)
                return model
            except Exception as inner_e:
                raise RuntimeError(
                    f"Failed to load model '{name}'. It's not registered in the registry "
                    f"and couldn't be loaded directly: {str(inner_e)}"
                ) from e
        
        # Check cache first
        if name in self._model_cache:
            return self._model_cache[name]
        
        # Prepare model arguments
        model_args = metadata.parameters.copy()
        model_args.update(kwargs)
        
        # Load config if not provided
        if config is None and metadata.config_class:
            config = metadata.config_class.from_pretrained(name, **model_args)
        
        # Load the model
        try:
            if metadata.model_class:
                model = metadata.model_class.from_pretrained(
                    name,
                    config=config,
                    **model_args
                )
            else:
                model = AutoModel.from_pretrained(
                    name,
                    config=config,
                    **model_args
                )
            
            # Move to device if specified
            if device is not None:
                model = model.to(device)
            
            # Cache the loaded model
            with self._lock:
                self._model_cache[name] = model
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{name}': {str(e)}") from e
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        with self._lock:
            self._model_cache.clear()


# Create a global instance of the registry
registry = ModelRegistry()

# Export common types
__all__ = [
    'ModelType',
    'ModelMetadata',
    'ModelRegistry',
    'registry'
]
