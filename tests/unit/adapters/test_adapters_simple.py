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

    # Mock/Augment transformers without replacing existing global mock
    t = sys.modules.get("transformers")
    if t is None:
        # Create a minimal transformers mock and mark as test env
        t = types.ModuleType("transformers")
        setattr(t, "MockTransformers", True)
        sys.modules["transformers"] = t

    # Ensure tokenization utils base with PreTrainedTokenizerBase exists
    tok_utils_name = "transformers.tokenization_utils_base"
    tok_utils = sys.modules.get(tok_utils_name)
    if tok_utils is None:
        tok_utils = types.ModuleType(tok_utils_name)
        sys.modules[tok_utils_name] = tok_utils
    if not hasattr(tok_utils, "PreTrainedTokenizerBase"):

        class MockPreTrainedTokenizerBase: ...

        tok_utils.PreTrainedTokenizerBase = MockPreTrainedTokenizerBase
    setattr(t, "tokenization_utils_base", tok_utils)

    # Provide PreTrainedModel for registry typing and adapters
    if not hasattr(t, "PreTrainedModel"):

        class MockPreTrainedModel: ...

        setattr(t, "PreTrainedModel", MockPreTrainedModel)

    # Provide Auto* classes used by adapters
    if not hasattr(t, "AutoModel"):

        class _AutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return MagicMock()

        setattr(t, "AutoModel", _AutoModel)
        setattr(t, "AutoModelForCausalLM", _AutoModel)

    if not hasattr(t, "AutoTokenizer"):

        class _AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                m = MagicMock()
                setattr(m, "pad_token_id", 0)
                return m

        setattr(t, "AutoTokenizer", _AutoTokenizer)

    if not hasattr(t, "AutoConfig"):

        class _AutoConfig:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return {}

        setattr(t, "AutoConfig", _AutoConfig)

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
