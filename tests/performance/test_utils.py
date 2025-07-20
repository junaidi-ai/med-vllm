"""Test utilities for performance testing."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, TypeVar
from unittest.mock import MagicMock


class ModelType(Enum):
    """Enumeration of model types."""

    GENERIC = auto()
    BIOMEDICAL = auto()
    CLINICAL = auto()


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    name: str
    model_type: ModelType
    model_class: Type
    config_class: Type
    description: str = ""
    tags: List[str] = None
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.parameters is None:
            self.parameters = {}


# Prevent pytest from collecting this as a test class
class TestModel:  # type: ignore
    """A simple test model class for performance testing."""

    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name", "test_model")
        self.config = MagicMock()
        self.config.model_type = "test_model"


# Prevent pytest from collecting this as a test class
class TestConfig:  # type: ignore
    """A simple test config class for performance testing."""

    def __init__(self, *args, **kwargs):
        self.model_type = "test_model"
        self.vocab_size = 10000
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
