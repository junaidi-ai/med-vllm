"""
Model Registry for managing and loading different model types.

This module provides a thread-safe registry for managing various model types,
including medical models like BioBERT and ClinicalBERT, with support for
model loading, caching, and metadata management.

Example:
    ```python
    # Get the registry instance
    registry = get_registry()

    # Register a new model
    registry.register(
        name="my-model",
        model_type=ModelType.BIOMEDICAL,
        model_class=AutoModel,
        config_class=AutoConfig,
        description="My custom biomedical model",
        tags=["custom", "biomedical"]
    )

    # Load a model
    model = registry.load_model("biobert-base-cased-v1.2")

    # List all registered models
    models = registry.list_models()
    ```

Thread Safety:
    All public methods are thread-safe. The registry uses a reentrant lock
    to ensure thread safety during concurrent access.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    ClassVar,
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
)

import torch
import yaml
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

# Import medical model loaders
try:
    from medvllm.models import BioBERTLoader, ClinicalBERTLoader
    from medvllm.models.medical_models import MedicalModelLoader

    MEDICAL_MODELS_AVAILABLE = True
except ImportError:
    MEDICAL_MODELS_AVAILABLE = False

from .exceptions import (
    ModelInitializationError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelRegistrationError,
    ModelRegistryError,
    ModelValidationError,
)
from .types import PretrainedConfigT

logger = logging.getLogger(__name__)

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

    This registry supports both standard and medical models, with specialized
    loading for medical models like BioBERT and ClinicalBERT.

    This class implements the singleton pattern to ensure there's only one registry instance.
    It provides methods for registering, retrieving, and managing models with their metadata.

    The registry is thread-safe and can be used in concurrent applications. All public methods
    are protected by a reentrant lock to ensure thread safety.

    Example:
        ```python
        from transformers import AutoModel, AutoConfig
        from medvllm.engine.model_runner.registry import get_registry
        from medvllm.engine.model_runner.types import ModelType

        # Get the registry instance
        registry = get_registry()

        # Register a model
        registry.register(
            name="example-model",
            model_type=ModelType.BIOMEDICAL,
            model_class=AutoModel,
            config_class=AutoConfig,
            description="Example model registration",
            tags=["example", "test"],
            # Additional parameters for model loading
            trust_remote_code=True,
            device_map="auto"
        )

        # Load the registered model
        model = registry.load_model("example-model")
        ```
    """

    _instance: ClassVar[Optional["ModelRegistry"]] = None
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _initialized: bool = False  # Track initialization state

    def __new__(cls) -> "ModelRegistry":
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
            self._loaders: Dict[str, Type[MedicalModelLoader]] = {}
            self._initialized = True
            self._register_default_models()

    def _validate_model_name(self, name: str) -> None:
        """Validate the model name.

        Args:
            name: The model name to validate.

        Raises:
            ModelValidationError: If the name is invalid.
        """
        if not name or not isinstance(name, str):
            raise ModelValidationError("Model name must be a non-empty string")

        if not all(c.isalnum() or c in ".-_" for c in name):
            raise ModelValidationError(
                "Model name can only contain alphanumeric characters, '.', '-', and '_'",
                model_name=name,
            )

    def _validate_model_type(self, model_type: ModelType) -> None:
        """Validate the model type.

        Args:
            model_type: The model type to validate.

        Raises:
            ModelValidationError: If the model type is invalid.
        """
        if not isinstance(model_type, ModelType):
            raise ModelValidationError(
                f"Invalid model type: {model_type}. Must be an instance of ModelType"
            )

    def _register_default_models(self) -> None:
        """Register default models with the registry.

        This method is called during initialization to register commonly used models.
        If registration of a default model fails, the error is logged but does not
        prevent the registry from being used.
        """
        try:
            # Register medical models if available
            if MEDICAL_MODELS_AVAILABLE:
                self._register_medical_models()

            # Load any additional model configurations
            self._load_model_configs()

        except Exception as e:
            logger.error(f"Failed to register default models: {e}", exc_info=True)

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
                logger.info("Registered default model: %s", model_info_typed["name"])
            except ModelRegistryError as e:
                logger.warning(
                    "Failed to register default model %s: %s",
                    model_info.get("name", "unknown"),
                    str(e),
                )

    def register(
        self,
        name: str,
        model_type: ModelType = ModelType.GENERIC,
        model_class: Optional[Type[M]] = None,
        config_class: Optional[Type[PretrainedConfig]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        loader: Optional[Type[MedicalModelLoader]] = None,
        **parameters: Any,
    ) -> None:
        """Register a new model with the registry.

        This method is thread-safe and will skip registration if a model with the same
        name already exists.

        Args:
            name: Unique name for the model. Must be a non-empty string containing only
                alphanumeric characters, '.', '-', or '_'.
            model_type: Type of the model (e.g., BIOMEDICAL, CLINICAL).
            model_class: The model class to use for instantiation. If None, AutoModel will be used.
            config_class: The config class to use for the model. If None, AutoConfig will be used.
            description: Optional description of the model.
            tags: Optional list of tags for categorizing the model.
            loader: Specialized loader for medical models.
            **parameters: Additional parameters to pass to the model during loading.

        Raises:
            ModelValidationError: If the model name or type is invalid.
            ModelRegistrationError: If there's an error during registration.

        Example:
            ```python
            registry = get_registry()
            registry.register(
                name="my-model",
                model_type=ModelType.BIOMEDICAL,
                model_class=AutoModel,
                config_class=AutoConfig,
                description="My custom model",
                tags=["custom", "biomedical"],
                trust_remote_code=True
            )
            ```
        """
        with self._lock:
            try:
                # Input validation
                self._validate_model_name(name)
                self._validate_model_type(model_type)

                # Skip if already registered
                if name in self._models:
                    logger.debug("Model '%s' is already registered, skipping", name)
                    return

                # Create and store model metadata
                params = parameters or {}
                if loader is not None:
                    params["loader"] = loader

                self._models[name] = ModelMetadata(
                    name=name,
                    model_type=model_type,
                    model_class=model_class,
                    config_class=config_class,
                    description=description,
                    tags=tags or [],
                    parameters=params,
                )

                logger.info("Registered model: %s (type: %s)", name, model_type.name)

            except ModelValidationError:
                raise  # Re-raise validation errors
            except Exception as e:
                raise ModelRegistrationError(
                    f"Failed to register model: {str(e)}", model_name=name, error=str(e)
                ) from e

    def unregister(self, name: str) -> None:
        """Unregister a model from the registry.

        Args:
            name: Name of the model to unregister.

        Raises:
            ModelNotFoundError: If the model is not found in the registry.
        """
        with self._lock:
            if name not in self._models:
                raise ModelNotFoundError(
                    f"Cannot unregister: model not found in registry", model_name=name
                )

            # Remove from cache if loaded
            if name in self._model_cache:
                del self._model_cache[name]
                logger.debug("Removed model '%s' from cache", name)

            # Remove from registry
            del self._models[name]
            logger.info("Unregistered model: %s", name)

    def get_metadata(self, name: str) -> ModelMetadata:
        """Get metadata for a registered model.

        Args:
            name: Name of the model.

        Returns:
            The model's metadata.

        Raises:
            ModelNotFoundError: If the model is not found in the registry.
        """
        with self._lock:
            if name not in self._models:
                raise ModelNotFoundError(
                    f"Model not found in registry", model_name=name
                )
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

    def _register_medical_models(self) -> None:
        """Register medical models with the registry.

        This method registers default medical models like BioBERT and ClinicalBERT
        if they are available in the environment.
        """
        if not MEDICAL_MODELS_AVAILABLE:
            logger.warning("Medical models are not available. Skipping registration.")
            return

        medical_models = [
            {
                "name": "biobert-base",
                "model_type": ModelType.BIOMEDICAL,
                "description": "BioBERT v1.1 - Biomedical Language Model",
                "tags": ["biomedical", "biobert", "pretrained"],
                "loader": BioBERTLoader,
                "parameters": {
                    "num_labels": 2,
                    "output_hidden_states": True,
                    "output_attentions": False,
                    "return_dict": True,
                    "torchscript": False,
                    "torch_dtype": "float32",
                    "use_cache": True,
                },
            },
            {
                "name": "clinical-bert-base",
                "model_type": ModelType.CLINICAL,
                "description": "Clinical BERT - Pretrained on clinical notes",
                "tags": ["clinical", "bert", "pretrained"],
                "loader": ClinicalBERTLoader,
                "parameters": {
                    "num_labels": 2,
                    "output_hidden_states": True,
                    "output_attentions": False,
                    "return_dict": True,
                    "torchscript": False,
                    "torch_dtype": "float32",
                    "use_cache": True,
                },
            },
        ]

        for model_info in medical_models:
            try:
                # Safely extract and convert parameters
                params: Dict[str, Any] = {}
                if "parameters" in model_info and isinstance(
                    model_info["parameters"], dict
                ):
                    params = {str(k): v for k, v in model_info["parameters"].items()}

                # Get loader if it exists
                loader = model_info.get("loader")

                # Create model metadata with type-safe conversions
                metadata = ModelMetadata(
                    name=str(model_info.get("name", "")),
                    model_type=ModelType(
                        model_info.get("model_type", ModelType.GENERIC)
                    ),
                    description=str(model_info.get("description", "")),
                    tags=(
                        [str(tag) for tag in model_info["tags"]]
                        if "tags" in model_info
                        and isinstance(model_info["tags"], (list, tuple, set))
                        else []
                    ),
                    parameters=params,
                )

                # Store the model metadata
                self._models[metadata.name] = metadata

                # Store the loader separately if provided and is the correct type
                if loader is not None and isinstance(loader, type):
                    self._loaders[metadata.name] = loader

                logger.info(
                    "Registered medical model: %s", model_info.get("name", "unknown")
                )

            except Exception as e:
                logger.warning(
                    "Failed to register medical model %s: %s",
                    model_info.get("name", "unknown"),
                    str(e),
                    exc_info=True,
                )

    def _load_model_configs(self) -> None:
        """Load model configurations from YAML files."""
        try:
            config_path = Path(__file__).parent.parent.parent / "configs" / "models"
            if not config_path.exists():
                return

            for config_file in config_path.glob("*.yaml"):
                try:
                    with open(config_file, "r") as f:
                        configs = yaml.safe_load(f) or {}

                    for model_id, model_info in configs.items():
                        try:
                            self.register(
                                name=model_info.get("name", model_id),
                                model_type=ModelType[
                                    model_info.get("model_type", "GENERIC").upper()
                                ],
                                description=model_info.get("description", ""),
                                tags=model_info.get("tags", []),
                                **model_info.get("parameters", {}),
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to load model config {model_id}: {e}"
                            )

                except Exception as e:
                    logger.warning(f"Failed to parse config file {config_file}: {e}")

        except Exception as e:
            logger.error(f"Error loading model configurations: {e}", exc_info=True)

    def _load_medical_model(
        self,
        name: str,
        metadata: ModelMetadata,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> M:
        """Load a medical model using its specialized loader.

        Args:
            name: Name of the model to load.
            metadata: The model's metadata.
            device: Device to load the model onto.
            **kwargs: Additional arguments for model loading.

        Returns:
            The loaded model.

        Raises:
            ModelLoadingError: If the model cannot be loaded.
        """
        if not MEDICAL_MODELS_AVAILABLE:
            raise ModelLoadingError(
                "Medical models are not available. Make sure to install the required dependencies.",
                model_name=name,
            )

        try:
            # Get the loader class from metadata
            loader_class = getattr(metadata, "loader", None)
            if not loader_class or not issubclass(loader_class, MedicalModelLoader):
                raise ModelLoadingError(
                    f"No valid loader found for medical model {name}", model_name=name
                )

            # Load the model using the specialized loader
            model, _ = loader_class.load_model(device=device, **kwargs)
            return model

        except Exception as e:
            raise ModelLoadingError(
                f"Failed to load medical model {name}: {str(e)}",
                model_name=name,
                error=str(e),
            ) from e

    def load_model(
        self,
        name: str,
        config: Optional[PretrainedConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> M:
        """Load a model by name.

        This method is thread-safe and supports loading both registered models and
        directly from the Hugging Face Hub. Loaded models are cached for future use.

        For medical models, it uses specialized loaders to handle model-specific
        configurations and requirements.

        Args:
            name: Name or path of the model to load. Can be a registered model name
                or a Hugging Face model identifier.
            config: Optional model configuration. If None, will be loaded from the
                registered config class or AutoConfig.
            device: Device to load the model onto.
            **kwargs: Additional arguments to pass to the model's from_pretrained method.

        Returns:
            The loaded model.

        Raises:
            ModelNotFoundError: If the model is not found in the registry or Hub.
            ModelLoadingError: If there's an error loading the model.
            ModelInitializationError: If the model cannot be initialized.

        Example:
            ```python
            # Load a registered medical model
            model = registry.load_model("biobert-base")

            # Load with custom device
            model = registry.load_model("clinical-bert-base", device="cuda:0")

            # Load with custom parameters
            model = registry.load_model("biobert-base", num_labels=3)
            ```
        """
        try:
            with self._lock:
                # Check if model is in cache first
                if name in self._model_cache:
                    return self._model_cache[name]

                # Try to get model metadata
                try:
                    metadata = self.get_metadata(name)

                    # Handle medical models with specialized loading
                    if metadata.model_type in (
                        ModelType.BIOMEDICAL,
                        ModelType.CLINICAL,
                    ):
                        model = self._load_medical_model(
                            name, metadata, device, **kwargs
                        )
                    else:
                        model = self._load_model_with_metadata(
                            name, metadata, config, device, kwargs
                        )

                    self._model_cache[name] = model
                    return model

                except ModelNotFoundError:
                    # If not found in registry, try direct loading
                    logger.debug(
                        "Model '%s' not found in registry, attempting direct load", name
                    )
                    return self._load_directly(name, device, **kwargs)  # type: ignore

            # Prepare model arguments
            model_args = metadata.parameters.copy()
            model_args.update(kwargs)

            # Load and return the model
            model = self._load_model_with_metadata(
                name, metadata, config, device, model_args
            )
            return model

        except ModelLoadingError:
            raise  # Re-raise loading errors
        except Exception as e:
            raise ModelLoadingError(
                f"Unexpected error loading model '{name}': {str(e)}",
                model_name=name,
                error=str(e),
            ) from e

    def _load_directly(
        self,
        name: str,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> M:
        """Load a model directly from Hugging Face Hub.

        Args:
            name: Model identifier from Hugging Face Hub.
            device: Device to load the model onto.
            **kwargs: Additional arguments for from_pretrained.

        Returns:
            The loaded model.

        Raises:
            ModelLoadingError: If the model cannot be loaded.
        """
        try:
            logger.info("Loading model directly from Hub: %s", name)
            model = AutoModel.from_pretrained(name, **kwargs)
            if device is not None:
                model = model.to(device)
            return model
        except Exception as e:
            raise ModelLoadingError(
                f"Failed to load model '{name}' from Hugging Face Hub: {str(e)}",
                model_name=name,
                error=str(e),
            ) from e

    def _load_model_with_metadata(
        self,
        name: str,
        metadata: ModelMetadata,
        config: Optional[PretrainedConfig],
        device: Optional[Union[str, torch.device]],
        model_args: Dict[str, Any],
    ) -> M:
        """Load a model using its metadata.

        Args:
            name: Name of the model to load.
            metadata: The model's metadata.
            config: Optional model configuration.
            device: Device to load the model onto.
            model_args: Additional model arguments.

        Returns:
            The loaded model.

        Raises:
            ModelInitializationError: If the model cannot be initialized.
        """
        try:
            # Load config if not provided
            if config is None and metadata.config_class:
                logger.debug("Loading config for model: %s", name)
                config = metadata.config_class.from_pretrained(name, **model_args)

            # Load the model
            logger.info("Loading model: %s", name)
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
            raise ModelInitializationError(
                f"Failed to initialize model '{name}': {str(e)}",
                model_name=name,
                error=str(e),
            ) from e

    def clear_cache(self) -> None:
        """Clear the model cache.

        This will force models to be reloaded on next access.
        """
        with self._lock:
            self._model_cache.clear()

    def clear(self) -> None:
        """Clear all models and cache from the registry.

        This is primarily useful for testing to ensure a clean state between tests.
        """
        with self._lock:
            self._models.clear()
            self._model_cache.clear()


def get_registry() -> "ModelRegistry[PreTrainedModel]":
    """Get or create the global model registry instance.

    This function ensures that the registry is properly initialized with default models
    and provides a single point of access to the registry instance.

    Returns:
        The global ModelRegistry instance.

    Example:
        ```python
        registry = get_registry()
        model = registry.load_model("biobert-base-cased-v1.2")
        ```
    """
    # This ensures thread-safe lazy initialization of the registry
    if ModelRegistry._instance is None or not hasattr(
        ModelRegistry._instance, "_initialized"
    ):
        with ModelRegistry._lock:
            if ModelRegistry._instance is None or not hasattr(
                ModelRegistry._instance, "_initialized"
            ):
                registry: ModelRegistry = ModelRegistry()
                registry._register_default_models()
    return cast("ModelRegistry[PreTrainedModel]", ModelRegistry._instance)


# Create a global instance of the registry for backward compatibility
# Prefer using get_registry() in new code
registry: ModelRegistry[PreTrainedModel] = get_registry()

# Export common types
__all__ = ["ModelType", "ModelMetadata", "ModelRegistry", "registry"]
