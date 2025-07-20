"""Shared test utilities and fixtures for the Med vLLM test suite.

This module provides common test utilities, fixtures, and mock objects that can be
used across different test modules. It's designed to help with testing components
that depend on external libraries like transformers and torch.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Re-export field mock for convenience
from tests.mock_field import field  # noqa: F401


class MockConfig:
    """Mock configuration class for transformers models."""

    def __init__(self, model_type="qwen3", **kwargs):
        self.model_type = model_type
        self.vocab_size = 32000
        self.hidden_size = 4096
        self.num_hidden_layers = 32
        self.num_attention_heads = 32
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class MockAutoConfig:
    """Mock for AutoConfig class."""

    @staticmethod
    def from_pretrained(model_name, **kwargs):
        """Mock implementation of from_pretrained."""
        if "qwen" in model_name.lower():
            return MockConfig(model_type="qwen3")
        return MockConfig(model_type=model_name.split("/")[-1])


class MockAutoModel:
    """Mock for AutoModel class."""

    @staticmethod
    def from_pretrained(model_name, **kwargs):
        """Mock implementation of from_pretrained."""
        model = MagicMock()
        model.config = MockAutoConfig.from_pretrained(model_name)
        return model


class MockAutoTokenizer:
    """Mock for AutoTokenizer class."""

    @staticmethod
    def from_pretrained(model_name, **kwargs):
        """Mock implementation of from_pretrained."""
        tokenizer = MagicMock()
        tokenizer.vocab_size = 32000
        tokenizer.model_max_length = 2048
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.bos_token_id = 2
        return tokenizer


class MockPreTrainedTokenizerBase:
    """Mock for PreTrainedTokenizerBase class."""

    def __init__(self, **kwargs):
        self.vocab_size = 32000
        self.model_max_length = 2048
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, *args, **kwargs):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


class MockVersions:
    """Mock for transformers.utils.versions module."""

    def require_version(self, *args, **kwargs):
        return True


class MockTorch:
    """Mock for torch module."""

    class cuda:
        """Mock for torch.cuda."""

        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 1

    class FloatTensor:
        """Mock for torch.FloatTensor."""

        def __new__(cls, *args, **kwargs):
            return MagicMock()

    class LongTensor:
        """Mock for torch.LongTensor."""

        def __new__(cls, *args, **kwargs):
            return MagicMock()

    class no_grad:
        """Mock for torch.no_grad."""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass


class CompleteTransformersMock:
    """Complete mock for the transformers module with all required submodules."""

    AutoConfig = MockAutoConfig
    AutoModel = MockAutoModel
    AutoTokenizer = MockAutoTokenizer
    PreTrainedTokenizerBase = MockPreTrainedTokenizerBase

    class utils:
        versions = MockVersions()

        class _pytree:
            @staticmethod
            def tree_map(fn, *args, **kwargs):
                return fn(*args)

        class generic:
            @staticmethod
            def ModelOutput(*args, **kwargs):
                class ModelOutput:
                    def __init__(self, *args, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)

                return ModelOutput(*args, **kwargs)

    class modeling_utils:
        class PreTrainedModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return MagicMock()

    class generation:
        class GenerationMixin:
            pass


@pytest.fixture(scope="function")
def mock_transformers():
    """Fixture that provides a complete mock of the transformers module."""
    # Store original modules
    orig_modules = {}
    for name in list(sys.modules.keys()):
        if name.startswith("transformers") or name == "torch":
            orig_modules[name] = sys.modules.pop(name, None)

    # Install our mocks
    transformers_mock = CompleteTransformersMock()
    torch_mock = MockTorch()

    # Patch sys.modules
    sys.modules["transformers"] = transformers_mock
    sys.modules["torch"] = torch_mock

    # Set up common submodules
    sys.modules.update(
        {
            "transformers.utils": transformers_mock.utils,
            "transformers.utils.versions": transformers_mock.utils.versions,
            "transformers.utils._pytree": transformers_mock.utils._pytree,
            "transformers.utils.generic": transformers_mock.utils.generic,
            "transformers.generation": transformers_mock.generation,
            "transformers.modeling_utils": transformers_mock.modeling_utils,
            "transformers.tokenization_utils_base": MockPreTrainedTokenizerBase(),
            "torch.cuda": torch_mock.cuda,
        }
    )

    yield {"transformers": transformers_mock, "torch": torch_mock}

    # Restore original modules
    for name, module in orig_modules.items():
        if module is not None:
            sys.modules[name] = module


@pytest.fixture(scope="function")
def model_registry(mock_transformers):
    """Fixture that provides a clean ModelRegistry instance with mocks."""
    from medvllm.engine.model_runner.registry import ModelRegistry

    # Clear any existing models from the registry
    registry = ModelRegistry()
    for model in registry.list_models():
        registry.unregister(model.name)

    return registry


@pytest.fixture(scope="function")
def mock_model_config():
    """Fixture that provides a mock model configuration."""
    return MockConfig()


@pytest.fixture(scope="function")
def mock_tokenizer():
    """Fixture that provides a mock tokenizer."""
    return MockAutoTokenizer.from_pretrained("test-model")
