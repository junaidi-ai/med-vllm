"""Custom exceptions for the model runner package."""

from typing import Any, Optional


class ModelRegistryError(Exception):
    """Base exception for model registry errors."""

    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs: Any):
        self.model_name = model_name
        self.details = kwargs
        if model_name:
            message = f"{message} (model: {model_name})"
        super().__init__(message)


class ModelRegistrationError(ModelRegistryError):
    """Raised when there's an error registering a model."""

    pass


class ModelLoadingError(ModelRegistryError):
    """Raised when there's an error loading a model."""

    pass


class ModelValidationError(ModelRegistryError):
    """Raised when model validation fails."""

    pass


class ModelNotFoundError(ModelRegistryError):
    """Raised when a requested model is not found in the registry."""

    pass


class ModelInitializationError(ModelRegistryError):
    """Raised when there's an error initializing a model."""

    pass
