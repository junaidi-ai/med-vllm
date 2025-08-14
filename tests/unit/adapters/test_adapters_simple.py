"""Simplified tests for medical model adapters with comprehensive mocks.

All mocking is isolated to this module and cleaned up after tests to avoid
polluting other tests.
"""

import sys
import types
import pytest
from unittest.mock import MagicMock, Mock


@pytest.fixture(autouse=True, scope="module")
def _isolated_torch_and_transformers_mocks():
    """Install local mocks for torch and transformers, then restore originals."""
    originals = {
        k: sys.modules.get(k)
        for k in [
            "torch",
            "torch.nn",
            "torch.nn.functional",
            "torch.optim",
            "torch.cuda",
            "torch.multiprocessing",
            "torch.distributed",
            "torch.nn.parameter",
            "transformers",
        ]
    }

    class MockTorch:
        class optim:
            class Optimizer:
                def __init__(self, *args, **kwargs):
                    pass

                def step(self, *args, **kwargs):
                    pass

                def zero_grad(self, *args, **kwargs):
                    pass

        class multiprocessing:
            class Process:
                def __init__(self, *args, **kwargs):
                    pass

                def start(self):
                    pass

                def join(self):
                    pass

        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

        @staticmethod
        def device(device_str):
            return device_str

    # Install torch mocks
    sys.modules["torch"] = MockTorch()
    sys.modules["torch.optim"] = MockTorch.optim
    sys.modules["torch.cuda"] = MockTorch.cuda
    sys.modules["torch.multiprocessing"] = MockTorch.multiprocessing
    sys.modules["torch.nn"] = Mock()
    sys.modules["torch.nn.functional"] = Mock()
    sys.modules["torch.nn.parameter"] = Mock(Parameter=Mock())
    sys.modules["torch.distributed"] = Mock()

    # Mock transformers
    mock_transformers = types.ModuleType("transformers")
    mock_tokenization_utils = types.ModuleType("transformers.tokenization_utils_base")

    class MockPreTrainedTokenizerBase: ...

    class MockPreTrainedModel: ...

    mock_tokenization_utils.PreTrainedTokenizerBase = MockPreTrainedTokenizerBase
    mock_transformers.tokenization_utils_base = mock_tokenization_utils
    mock_transformers.PreTrainedModel = MockPreTrainedModel

    class MockAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return MagicMock()

    class MockAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return MagicMock()

    class MockAutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return {}

    mock_transformers.AutoModel = MockAutoModel
    mock_transformers.AutoModelForCausalLM = MockAutoModel
    mock_transformers.AutoTokenizer = MockAutoTokenizer
    mock_transformers.AutoConfig = MockAutoConfig
    sys.modules["transformers"] = mock_transformers

    try:
        yield
    finally:
        # Restore originals
        for k, v in originals.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)


# Import after mocks are installed by the fixture
from medvllm.models.adapters.base import MedicalModelAdapterBase
from medvllm.models.adapters.biobert import BioBERTAdapter
from medvllm.models.adapters.clinicalbert import ClinicalBERTAdapter

# Alias for backward compatibility
MedicalModelAdapter = MedicalModelAdapterBase


# Test cases
def test_medical_model_adapter_initialization():
    """Test basic initialization of MedicalModelAdapterBase."""
    mock_model = MagicMock()
    mock_model.config = {}

    adapter = MedicalModelAdapterBase(model=mock_model, config={"model_type": "test"})
    assert adapter.model == mock_model
    assert adapter.config["model_type"] == "test"


def test_biobert_adapter_initialization():
    """Test initialization of BioBERT adapter."""
    mock_model = MagicMock()
    mock_model.config = {}

    adapter = BioBERTAdapter(model=mock_model, config={"model_type": "biobert"})
    assert isinstance(adapter, MedicalModelAdapterBase)
    assert adapter.config["model_type"] == "biobert"


def test_clinicalbert_adapter_initialization():
    """Test initialization of ClinicalBERT adapter."""
    mock_model = MagicMock()
    mock_model.config = {}

    adapter = ClinicalBERTAdapter(model=mock_model, config={"model_type": "clinicalbert"})
    assert isinstance(adapter, MedicalModelAdapterBase)
    assert adapter.config["model_type"] == "clinicalbert"
