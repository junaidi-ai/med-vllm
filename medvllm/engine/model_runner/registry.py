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
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from .types import PretrainedConfigT

# Type variable for model types
M = TypeVar("M", bound=PreTrainedModel)


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
            "name": self.name,
            "model_type": self.model_type.name,
            "model_class": self.model_class.__name__ if self.model_class else None,
            "config_class": self.config_class.__name__ if self.config_class else None,
            "description": self.description,
            "tags": self.tags,
            "parameters": self.parameters,
        }


class ModelRegistry(Generic[M]):
    """
    A thread-safe registry for managing model loading and instantiation.

    This class implements the singleton pattern to ensure there's only one registry instance.
    It provides methods for registering, retrieving, and managing models with their metadata.
    """

    _instance: ClassVar[Optional[ModelRegistry]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _initialized: bool = False  # Track initialization state

    def __new__(cls) -> ModelRegistry:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized") or not self._initialized:
            self._models: Dict[str, ModelMetadata] = {}
            self._model_cache: Dict[str, M] = {}
            self._initialized = True
            self._register_default_models()

    def _register_default_models(self) -> None:
        """Register default models with the registry."""
        default_models = [
            {
                "name": "biobert-base-cased-v1.2",
                "model_type": ModelType.BIOMEDICAL,
                "description": "BioBERT v1.2 - Biomedical Language Model",
                "tags": ["biomedical", "biobert", "pretrained"],
            },
            {
                "name": "clinical-bert-base-uncased",
                "model_type": ModelType.CLINICAL,
                "description": "Clinical BERT - Pretrained on clinical notes",
                "tags": ["clinical", "bert", "pretrained"],
            },
        ]

        for model_info in default_models:
            try:
                # Explicitly type the model_info dict to help mypy
                model_info_typed: Dict[str, Any] = model_info
                self.register(
                    name=model_info_typed["name"],
                    model_type=model_info_typed["model_type"],
                    description=model_info_typed["description"],
                    tags=model_info_typed["tags"],
                )
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to register default model {model_info['name']}: {e}"
                )

    def register(
        self,
        name: str,
        model_type: ModelType = ModelType.GENERIC,
        model_class: Optional[Type[M]] = None,
        config_class: Optional[Type[PretrainedConfig]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        **parameters: Any,
    ) -> None:
        """Register a new model with the registry.

        Args:
            name: Unique name for the model.
            model_type: Type of the model (e.g., BIOMEDICAL, CLINICAL).
            model_class: The model class to use for instantiation.
            config_class: The config class to use for the model.
            description: Optional description of the model.
            tags: Optional list of tags for the model.
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
                parameters=parameters or {},
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

    def list_models(
        self, model_type: Optional[ModelType] = None
    ) -> List[Dict[str, Any]]:
        """List all registered models, optionally filtered by type.

        Args:
            model_type: If provided, only return models of this type.

        Returns:
            List of dictionaries containing model metadata.
        """
        with self._lock:
            if model_type is None:
                return [model.to_dict() for model in self._models.values()]
            return [
                model.to_dict()
                for model in self._models.values()
                if model.model_type == model_type
            ]

    def is_registered(self, name: str) -> bool:
        """Check if a model is registered.

        Args:
            name: Name of the model to check.

        Returns:
            True if the model is registered, False otherwise.
        """
        with self._lock:
            return name in self._models

    def load_model(
        self,
        name: str,
        config: Optional[PretrainedConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> M:
        """Load a model by name.

        Args:
            name: Name of the model to load.
            config_or_device: Either a model configuration or a device specification.
            device: Device to load the model onto. If None, will use the default device.
            **kwargs: Additional arguments to pass to the model's from_pretrained method.

        Returns:
            The loaded model instance.

        Raises:
            KeyError: If the model is not found in the registry.
            RuntimeError: If there's an error loading the model.
        """
        # Handle the case where the second argument is a device (for backward compatibility)
        if (
            config is not None
            and not isinstance(config, PretrainedConfig)
            and device is None
        ):
            # If the second argument is not a PretrainedConfig and device is None,
            # assume it's a device
            device = config  # type: ignore[assignment]
            config = None
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
                    name, config=config, **model_args
                )
            else:
                model = AutoModel.from_pretrained(name, config=config, **model_args)

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
        """Clear the model cache.

        This will force models to be reloaded on next access.
        """
        with self._lock:
            self._model_cache.clear()


# Create a global instance of the registry
registry: ModelRegistry[PreTrainedModel] = ModelRegistry()

# Register default models
registry._register_default_models()

# Export common types
__all__ = ["ModelType", "ModelMetadata", "ModelRegistry", "registry"]
