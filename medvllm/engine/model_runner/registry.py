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
import time
from collections import OrderedDict as OrderedDictType
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BertConfig,
    BertModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig as _PretrainedConfig
from transformers.modeling_utils import PreTrainedModel as _PreTrainedModel

# Import TypeVar from typing_extensions first to avoid conflicts
from typing_extensions import TypeVar

from medvllm.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)
from medvllm.engine.model_runner.exceptions import (
    ModelInitializationError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelRegistrationError,
    ModelRegistryError,
    ModelValidationError,
)

# Import medical model loaders with type checking
MEDICAL_MODELS_AVAILABLE: bool = False

# Define type variables at module level
if TYPE_CHECKING:
    from torch import nn
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )
    from transformers.configuration_utils import PretrainedConfig
    from typing_extensions import TypeGuard, TypeVar

    from medvllm.models.adapters.medical_adapter_base import MedicalModelAdapterBase

    # Type variable for model class
    ModelT = TypeVar("ModelT", bound=PreTrainedModel)
    TokenizerT = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    ConfigT = TypeVar("ConfigT", bound=PretrainedConfig)
    LayerT = TypeVar("LayerT", bound=nn.Module)

    # Type aliases for medical model adapters and loaders
    _BioBERTLoader = Type[MedicalModelAdapterBase]
    _ClinicalBERTLoader = Type[MedicalModelAdapterBase]
    _MedicalModelLoader = Type[MedicalModelAdapterBase]
else:
    # Define placeholders for runtime
    from typing import Any, Type, TypeVar

    ModelT = TypeVar("ModelT", bound=Any)
    TokenizerT = Any
    ConfigT = TypeVar("ConfigT")
    LayerT = TypeVar("LayerT")
    _BioBERTLoader = Any
    _ClinicalBERTLoader = Any
    _MedicalModelLoader = Any

# Type alias for the model cache type (timestamp, model)
ModelCacheType = Dict[str, Tuple[float, ModelT]]


class ModelType(Enum):
    """Supported model types in the registry."""

    GENERIC = auto()
    BIOMEDICAL = auto()
    CLINICAL = auto()


@dataclass
class ModelMetadata:
    """Metadata for a registered model with enhanced metadata tracking."""

    name: str
    model_type: ModelType = ModelType.GENERIC
    model_class: Optional[Type[PreTrainedModel]] = None
    config_class: Optional[Type[PretrainedConfig]] = None
    tokenizer_class: Optional[Type[Any]] = None
    description: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    load_count: int = 0
    last_loaded: Optional[datetime] = None
    load_durations: Deque[float] = field(default_factory=deque)
    avg_load_time: float = 0.0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type.name,
            "model_class": self.model_class.__name__ if self.model_class else None,
            "config_class": self.config_class.__name__ if self.config_class else None,
            "tokenizer_class": (
                self.tokenizer_class.__name__ if self.tokenizer_class else None
            ),
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "parameters": self.parameters,
            "load_count": self.load_count,
            "last_loaded": self.last_loaded.isoformat() if self.last_loaded else None,
            "avg_load_time": self.avg_load_time,
            "capabilities": self.capabilities,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def update_load_metrics(self, load_duration: float) -> None:
        """Update performance metrics after a model load."""
        self.load_count += 1
        self.last_loaded = datetime.utcnow()
        self.load_durations.append(load_duration)


# Define the base model registry class with proper type parameters
class ModelRegistry(Generic[ModelT, ConfigT]):
    """Thread-safe registry for managing and loading different model types.

    This class implements a singleton pattern to ensure there's only one registry instance.
    It provides methods for registering, loading, and managing models with their metadata.

    The registry is thread-safe and can be used in concurrent applications. All public methods
    are protected by a reentrant lock to ensure thread safety.
    """

    _instance: ClassVar[Optional[ModelRegistry[Any, Any]]] = None
    _class_lock: ClassVar[threading.RLock] = threading.RLock()
    _initialized: bool = False

    # Default cache settings
    DEFAULT_CACHE_SIZE: int = 5
    CACHE_TTL: int = 3600  # 1 hour in seconds

    # Instance variables with type hints
    _models: Dict[str, ModelMetadata]
    _model_cache: Dict[str, Tuple[float, Any]]
    _loaders: Dict[str, Type[Any]]
    _cache_hits: int
    _cache_misses: int
    _cache_size: int

    def __new__(cls) -> "ModelRegistry[ModelT, ConfigT]":
        with cls._class_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance  # type: ignore

    def __init__(self) -> None:
        # Only initialize if not already initialized
        if not self._initialized:
            self._models: Dict[str, ModelMetadata] = {}
            self._model_cache: Dict[str, Tuple[float, Any]] = {}
            self._loaders: Dict[str, Type[Any]] = {}
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_size = self.DEFAULT_CACHE_SIZE
            self._lock = threading.RLock()  # Instance-specific lock
            self._initialized = True

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

    def _register_default_models(self, force: bool = False) -> None:
        """Register default models with the registry.

        This method is called during initialization to register commonly used models.
        If registration of a default model fails, the error is logged but does not
        prevent the registry from being used.
        """
        try:
            # Import the specific model classes to avoid type issues
            from transformers import BertConfig, BertModel

            # Register generic BERT model with specific model class
            self.register(
                name="bert-base-uncased",
                model_type=ModelType.GENERIC,
                model_class=BertModel,  # type: ignore[arg-type]  # Use concrete model class instead of AutoModel
                config_class=BertConfig,  # type: ignore[arg-type]  # Use concrete config class instead of AutoConfig
                description="Standard BERT model with uncased vocabulary",
                tags=["generic", "bert"],
                force=force,
            )

            # Register medical models if available
            self._register_medical_models()

        except Exception as e:
            logger.warning(f"Failed to register default models: {e}")
            if __debug__:
                import traceback

                logger.debug(traceback.format_exc())

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
        model_type: ModelType,
        model_class: Optional[Type[PreTrainedModel]] = None,
        config_class: Optional[Type[PretrainedConfig]] = None,
        tokenizer_class: Optional[Type[Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        loader: Optional[Type[Any]] = None,  # type: ignore[valid-type]
        parameters: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        """Register a new model with the registry.

        This method is thread-safe and will raise a ValueError if a model with the same
        name is already registered.

        Args:
            name: Unique name for the model. Must be a non-empty string containing only
                alphanumeric characters, hyphens, or underscores.
            model_type: Type of the model (e.g., BIOMEDICAL, CLINICAL).
            model_class: The model class to register. If None, will try to auto-detect.
            config_class: The config class for the model. If None, will try to auto-detect.
            tokenizer_class: The tokenizer class for the model. If None, will try to auto-detect.
            description: Optional description of the model.
            tags: Optional list of tags for the model.
            loader: Optional custom loader class for the model.
            parameters: Optional parameters to pass to the model during loading.
            **parameters: Additional parameters to pass to the model during loading.

        Raises:
            ValueError: If the model name is invalid, required parameters are missing,
                       or a model with the same name already exists.
            ModelRegistrationError: If there's an error during model registration.
        """
        with self._lock:
            try:
                # Validate input parameters
                if not name or not isinstance(name, str):
                    raise ValueError("Model name must be a non-empty string")

                # Check if model already exists
                if name in self._models:
                    if not force:
                        raise ValueError(
                            f"Model with name '{name}' is already registered"
                        )
                    del self._models[name]

                # Create and store model metadata
                metadata = ModelMetadata(
                    name=name,
                    model_type=model_type,
                    model_class=model_class,
                    config_class=config_class,
                    tokenizer_class=tokenizer_class,
                    description=description,
                    tags=tags or [],
                    parameters=parameters or {},
                )

                # Store the metadata
                self._models[name] = metadata

                # Store the loader if provided
                if loader is not None and hasattr(self, "_loaders"):
                    self._loaders[name] = loader

                logger.info(f"Registered model: {name}")

            except ValueError as e:
                logger.error(f"Validation error registering model {name}: {e}")
                raise  # Re-raise ValueError directly
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
        """Register medical models if the required dependencies are available."""
        if not MEDICAL_MODELS_AVAILABLE:
            return

        try:
            # Import at runtime to avoid circular imports
            from transformers import BertConfig, BertModel

            from medvllm.models.adapters.biobert_adapter import BioBERTAdapter
            from medvllm.models.adapters.clinicalbert_adapter import ClinicalBERTAdapter

            # Register BioBERT
            self.register(
                name="dmis-lab/biobert-v1.1",
                model_type=ModelType.BIOMEDICAL,
                model_class=BertModel,  # type: ignore[arg-type]
                config_class=BertConfig,  # type: ignore[arg-type]
                loader=BioBERTAdapter,  # type: ignore[arg-type]
                description="BioBERT: Biomedical Text Mining with BERT",
                tags=["biomedical", "bert"],
            )

            # Register ClinicalBERT
            self.register(
                name="emilyalsentzer/Bio_ClinicalBERT",
                model_type=ModelType.CLINICAL,
                model_class=BertModel,  # type: ignore[arg-type]
                config_class=BertConfig,  # type: ignore[arg-type]
                loader=ClinicalBERTAdapter,  # type: ignore[arg-type]
                description="ClinicalBERT: Clinical Text Mining with BERT",
                tags=["clinical", "bert"],
            )

        except ImportError as e:
            logger.warning(f"Failed to import medical model adapters: {e}")
        except Exception as e:
            logger.warning(f"Failed to register medical models: {e}")

    def _get_medical_loader(
        self, model_type: Union[str, ModelType]
    ) -> Optional[Type[Any]]:  # Using Type[Any] to avoid circular imports
        """Get the medical loader for a given model type.

        Args:
            model_type: The model type to get the loader for.

        Returns:
            The medical loader class for the given model type, or None if not found.
        """
        if isinstance(model_type, str):
            model_type = ModelType[model_type.upper()]

        loader = self._loaders.get(model_type.name.lower())
        return loader

    def _load_medical_model(
        self,
        name: str,
        metadata: Any,  # Using Any to avoid circular imports, will validate required attributes
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> ModelT:
        """Load a medical model using the appropriate adapter.

        Args:
            name: The name of the model to load.
            metadata: The model metadata.
            device: The device to load the model on.
            **kwargs: Additional keyword arguments to pass to the model loader.

        Returns:
            The loaded model.

        Raises:
            ModelLoadingError: If there is an error loading the model.
        """
        from torch.nn import Module
        from transformers import AutoConfig

        from medvllm.models.adapters.medical_adapter_base import MedicalModelAdapterBase

        # Get the model type from metadata or infer from name
        model_type = metadata.model_type or name.split("-")[0].lower()
        loader_class = self._get_medical_loader(model_type)

        if loader_class is None:
            raise ModelLoadingError(
                f"No registered loader for model type: {model_type}"
            )

        # Validate loader class type
        if not isinstance(loader_class, type) or not issubclass(
            loader_class, MedicalModelAdapterBase
        ):
            raise ModelLoadingError(
                f"Invalid loader class for model type {model_type}. "
                f"Expected a subclass of MedicalModelAdapterBase, got {type(loader_class).__name__}"
            )

        try:
            # Load config if not provided
            config: Dict[str, Any] = {}
            if hasattr(metadata, "config") and metadata.config is not None:
                config = metadata.config
            elif hasattr(metadata, "config_path") and metadata.config_path:
                config = AutoConfig.from_pretrained(metadata.config_path)

            # Initialize loader with model=None and config
            loader = loader_class(model=None, config=config)  # type: ignore[call-arg]

            # Load the model using the loader
            model = loader.load_model(name, **kwargs)

            # Move model to device if specified and model is a torch Module
            if device is not None and isinstance(model, Module):
                try:
                    model = model.to(device)
                except Exception as e:
                    raise ModelLoadingError(
                        f"Failed to move model to device {device}: {str(e)}"
                    ) from e

            return cast(ModelT, model)

        except ImportError as ie:
            raise ModelLoadingError(
                f"Missing dependencies for medical model {name}: {str(ie)}",
                model_name=name,
                error=str(ie),
            ) from ie
        except Exception as e:
            error_msg = f"Failed to load medical model '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ModelLoadingError(error_msg, model_name=name) from e

    def load_model(
        self,
        name: str,
        config: Optional[PretrainedConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> ModelT:
        """Load a model by name with enhanced caching and metadata tracking.

        This method is thread-safe and supports loading both registered models and
        directly from the Hugging Face Hub. Loaded models are cached for future use.
        For medical models, it uses specialized loaders to handle model-specific
        configurations and requirements.

        Type Safety:
            - Ensures proper type annotations for all parameters and return values
            - Validates model types against expected interfaces
            - Provides detailed type checking for model configurations

        Thread Safety:
            - Uses reentrant locks to ensure thread-safe model loading and caching
            - Implements LRU cache eviction policy
            - Handles concurrent access to shared resources

        Args:
            name: Name or path of the model to load. Can be a registered model name
                or a Hugging Face model identifier.
            config: Optional model configuration. If None, will be loaded from the
                registered config class or AutoConfig.
            device: Device to load the model onto. Can be a string ('cuda', 'cpu')
                or a torch.device object. If None, uses the default device.
            use_cache: Whether to use the model cache. Defaults to True.
                When disabled, forces a fresh load of the model.
            **kwargs: Additional arguments to pass to the model's from_pretrained method.
                Common arguments include:
                - force_download: bool = False
                - resume_download: bool = False
                - local_files_only: bool = False
                - use_auth_token: Optional[Union[bool, str]] = None

        Returns:
            The loaded model instance, which is a subclass of PreTrainedModel.

        Raises:
            ModelNotFoundError: If the model is not found in the registry or Hub.
            ModelLoadingError: If there's an error loading the model.
            ModelInitializationError: If the model cannot be initialized.
            ValueError: If the model name is invalid or configuration is invalid.
            RuntimeError: If there's an error during model loading or initialization.

        Example:
            ```python
            # Load a model with default settings
            model = registry.load_model("bert-base-uncased")

            # Load a model with custom device and configuration
            config = AutoConfig.from_pretrained("bert-base-uncased")
            model = registry.load_model(
                "bert-base-uncased",
                config=config,
                device="cuda:0",
                force_download=True
            )
            ```
        """
        if not name:
            raise ValueError("Model name cannot be empty")

        logger.debug(
            f"Loading model '{name}' with config: {config}, device: {device}, use_cache: {use_cache}"
        )

        # Convert device string to torch.device if needed
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except RuntimeError as e:
                raise ValueError(f"Invalid device string '{device}': {str(e)}") from e

        # Initialize model arguments with defaults if not provided
        model_args: Dict[str, Any] = {}
        if config is not None:
            model_args["config"] = config
        if device is not None:
            model_args["device_map"] = device

        # Check cache first if enabled
        if use_cache:
            with self._lock:
                if name in self._model_cache:
                    cached_item = self._model_cache[name]
                    if cached_item is not None:
                        cached_model, timestamp = cached_item
                        # Check if cache entry is still valid (within TTL)
                        if time.time() - timestamp < self.CACHE_TTL:
                            self._cache_hits += 1
                            logger.debug(f"Cache hit for model '{name}'")
                            return cast(ModelT, cached_model)
                        else:
                            logger.debug(
                                f"Cache entry for model '{name}' expired, reloading"
                            )
                            self._model_cache.pop(name, None)
            self._cache_misses += 1

        # Acquire lock to ensure thread safety during model loading
        with self._lock:
            start_time = time.time()
            model: Optional[ModelT] = None

            try:
                # Check if the model is registered
                if name in self._models:
                    metadata = self._models[name]
                    logger.debug(
                        f"Loading registered model '{name}' with metadata: {metadata}"
                    )

                    # Handle medical models with specialized loaders
                    if metadata.model_type in (
                        ModelType.BIOMEDICAL,
                        ModelType.CLINICAL,
                    ):
                        try:
                            load_start = time.time()
                            model = self._load_medical_model(
                                name, metadata, device, **kwargs
                            )
                            load_duration = time.time() - load_start
                            metadata.update_load_metrics(load_duration)

                            # Cache the loaded model if caching is enabled and model was loaded successfully
                            if use_cache and model is not None:
                                self._model_cache[name] = (time.time(), model)

                            return model
                        except Exception as load_error:
                            error_msg = f"Failed to load medical model '{name}': {str(load_error)}"
                            logger.error(error_msg, exc_info=True)
                            raise ModelLoadingError(
                                error_msg, model_name=name, error=str(load_error)
                            ) from load_error
                    else:
                        # For non-medical models, use standard loading with metadata
                        model_args.update(metadata.parameters or {})
                        model_args.update(kwargs)

                        load_start = time.time()
                        try:
                            model = self._load_model_with_metadata(
                                name, metadata, config, device, model_args
                            )
                            load_duration = time.time() - load_start
                            metadata.update_load_metrics(load_duration)

                            # Cache the loaded model if caching is enabled and model was loaded successfully
                            if use_cache and model is not None:
                                self._model_cache[name] = (time.time(), model)

                            return model

                        except Exception as load_error:
                            error_msg = (
                                f"Failed to load model '{name}': {str(load_error)}"
                            )
                            logger.error(error_msg, exc_info=True)
                            raise ModelLoadingError(
                                error_msg, model_name=name, error=str(load_error)
                            ) from load_error
                else:
                    # Try to load directly from Hugging Face Hub if not found in registry
                    try:
                        load_start = time.time()
                        model = self._load_directly(name, device, **kwargs)
                        load_duration = time.time() - load_start

                        # Create metadata for direct loads if not exists
                        if name not in self._models:
                            self._models[name] = ModelMetadata(
                                name=name,
                                model_type=ModelType.GENERIC,
                                description=f"Auto-loaded model: {name}",
                            )

                        # Update load metrics
                        self._models[name].update_load_metrics(load_duration)

                        # Cache the loaded model if caching is enabled and model was loaded successfully
                        if use_cache and model is not None:
                            self._model_cache[name] = (time.time(), model)

                        return model

                    except Exception as hub_error:
                        error_msg = f"Failed to load model '{name}' from Hugging Face Hub: {str(hub_error)}"
                        logger.error(error_msg, exc_info=True)
                        raise ModelLoadingError(
                            error_msg, model_name=name, error=str(hub_error)
                        ) from hub_error

            except Exception as e:
                error_msg = f"Unexpected error loading model '{name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ModelLoadingError(error_msg, model_name=name, error=str(e)) from e

            finally:
                # Ensure we always release resources if something went wrong
                if model is None:
                    logger.warning(f"Failed to load model: {name}")
                else:
                    logger.debug(
                        f"Successfully loaded model '{name}' in {time.time() - start_time:.2f}s"
                    )

            return model  # type: ignore

    def _load_directly(
        self,
        name: str,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> ModelT:
        """Load a model directly from Hugging Face Hub.

        Args:
            name: The name or path of the model to load.
            device: The device to load the model onto.
            **kwargs: Additional arguments to pass to the model's from_pretrained method.

        Returns:
            The loaded model.

        Raises:
            ModelLoadingError: If the model cannot be loaded.
        """
        try:
            logger.info(f"Loading model directly from Hub: {name}")
            model = AutoModel.from_pretrained(name, **kwargs)
            if device is not None:
                model = model.to(device)
            return cast(ModelT, model)
        except Exception as e:
            error_msg = f"Failed to load model '{name}' from Hugging Face Hub: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ModelLoadingError(error_msg, model_name=name, error=str(e)) from e

    def _load_model_with_metadata(
        self,
        name: str,
        metadata: ModelMetadata,
        config: Optional[PretrainedConfig],
        device: Optional[Union[str, torch.device]],
        model_args: Dict[str, Any],
    ) -> ModelT:
        """Load a model using its metadata.

        Args:
            name: Name of the model to load.
            metadata: The model's metadata.
            config: Optional model configuration.
            device: Device to load the model onto.
            model_args: Additional model arguments.

        Returns:
            The loaded model instance.

        Raises:
            ModelInitializationError: If the model cannot be initialized.
        """
        try:
            # Load config if not provided
            if config is None and metadata.config_class:
                config = metadata.config_class.from_pretrained(name, **model_args)
            elif config is None:
                config = AutoConfig.from_pretrained(name, **model_args)

            # Handle medical models with specialized loaders
            if metadata.model_type in (ModelType.BIOMEDICAL, ModelType.CLINICAL):
                loader_class = self._get_medical_loader(metadata.model_type)
                if loader_class is not None:
                    loader = loader_class(model=None, config=config)
                    loaded_model = loader.load_model(name, **model_args)
                    if device is not None:
                        loaded_model = loaded_model.to(device)
                    return cast(ModelT, loaded_model)

            # Load standard model with type safety
            model: ModelT
            if metadata.model_class:
                model = cast(
                    ModelT,
                    metadata.model_class.from_pretrained(
                        name, config=config, **model_args
                    ),
                )
            else:
                model = cast(
                    ModelT, AutoModel.from_pretrained(name, config=config, **model_args)
                )

            # Move to device if specified
            if device is not None:
                model = model.to(device)

            return model

        except Exception as e:
            error_msg = f"Failed to load model '{name}' with metadata: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ModelInitializationError(error_msg) from e

    def clear_cache(self) -> None:
        """Clear the model cache.

        This will force models to be reloaded on next access.
        Also resets cache statistics.
        """
        with self._lock:
            self._model_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("Model cache cleared and statistics reset")

    def set_cache_size(self, max_size: int) -> None:
        """Set the maximum number of models to keep in the cache.

        Args:
            max_size: Maximum number of models to cache. Must be >= 1.

        Raises:
            ValueError: If max_size is less than 1.
        """
        if max_size < 1:
            raise ValueError("Cache size must be at least 1")

        with self._lock:
            self._cache_size = max_size
            # If new size is smaller, evict oldest entries
            while len(self._model_cache) > self._cache_size:
                oldest_key = next(iter(self._model_cache))
                self._model_cache.pop(oldest_key)

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set the time-to-live for cache entries in seconds.

        Args:
            ttl_seconds: Time in seconds before a cache entry expires.

        Raises:
            ValueError: If ttl_seconds is less than 0.
        """
        if ttl_seconds < 0:
            raise ValueError("TTL must be >= 0 seconds")

        with self._lock:
            ModelRegistry.CACHE_TTL = ttl_seconds

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics including hit rate, size, etc.
        """
        with self._lock:
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total if total > 0 else 0.0

            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": hit_rate,
                "current_size": len(self._model_cache),
                "max_size": self._cache_size,
                "ttl_seconds": self.CACHE_TTL,
                "cached_models": list(self._model_cache.keys()),
            }

    def cache_hit_rate(self) -> float:
        """Calculate the current cache hit rate.

        Returns:
            The cache hit rate as a float between 0 and 1.
        """
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """Clear all models and cache from the registry.

        This is primarily useful for testing to ensure a clean state between tests.
        """
        with self._lock:
            self._models.clear()
            self._model_cache.clear()


def get_registry() -> "ModelRegistry[PreTrainedModel, PretrainedConfig]":
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
    if ModelRegistry._instance is None:
        with ModelRegistry._class_lock:
            if ModelRegistry._instance is None:
                # Create a new instance with explicit type parameters
                registry: ModelRegistry[PreTrainedModel, PretrainedConfig] = (
                    ModelRegistry()
                )
                registry._register_default_models()
                # _initialized is now handled in __new__
                ModelRegistry._instance = registry
    return cast(
        ModelRegistry[PreTrainedModel, PretrainedConfig], ModelRegistry._instance
    )


# Create a global instance of the registry for backward compatibility
# Prefer using get_registry() in new code
registry: "ModelRegistry[PreTrainedModel, PretrainedConfig]" = get_registry()

# Export common types
__all__ = ["ModelType", "ModelMetadata", "ModelRegistry", "registry"]
