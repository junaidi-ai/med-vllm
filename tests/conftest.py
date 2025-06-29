"""Pytest configuration and fixtures."""

import contextlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
from unittest.mock import MagicMock, Mock

# Import torch for type checking and mocking
import torch

import pytest

# Import mock field function to handle description parameter
from tests.mock_field import field  # noqa: F401

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock flash_attn and other CUDA-related modules
class MockFlashAttn:
    @staticmethod
    def flash_attn_varlen_func(*args, **kwargs):
        return MagicMock()

    @staticmethod
    def flash_attn_with_kvcache(*args, **kwargs):
        return MagicMock()

# Create a proper module mock that works with importlib
class ModuleMock(Mock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__spec__ = Mock()
        self.__spec__.name = self.__class__.__name__.lower()
        self.__path__ = []
        self.__package__ = ''
        self.__version__ = '2.0.0'

# Mock torch and CUDA-related modules
class MockTorch(ModuleMock):
    class cuda(ModuleMock):
        @staticmethod
        def is_available():
            return False
            
        class Stream(ModuleMock):
            pass
            
    class nn(ModuleMock):
        class Parameter(ModuleMock):
            def __new__(cls, data=None, requires_grad=True):
                instance = super().__new__(cls)
                instance.data = data if data is not None else MockTorch.Tensor()
                instance.requires_grad = requires_grad
                instance.grad = None
                return instance
                
            def __init__(self, data=None, requires_grad=True):
                super().__init__()
                self.data = data if data is not None else MockTorch.Tensor()
                self.requires_grad = requires_grad
                self.grad = None
                
            def __repr__(self):
                return f"Parameter containing:\n{self.data}"
                
        class Module(ModuleMock):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.training = True
                self._parameters = {}
                self._buffers = {}
                self._modules = {}
                
            def register_parameter(self, name, param):
                self._parameters[name] = param
                return param
                
            def register_buffer(self, name, tensor):
                self._buffers[name] = tensor
                return tensor
                
            def to(self, *args, **kwargs):
                return self
                
            def cuda(self, *args, **kwargs):
                return self
                
            def eval(self, *args, **kwargs):
                self.training = False
                return self
                
            def train(self, mode=True):
                self.training = mode
                return self
                
            def forward(self, *args, **kwargs):
                return args[0] if args else None
                
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        class Linear(ModuleMock):
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = MockTorch.nn.Parameter(MockTorch.Tensor())
                if bias:
                    self.bias = MockTorch.nn.Parameter(MockTorch.Tensor())
                else:
                    self.register_parameter('bias', None)
                    
        class functional(ModuleMock):
            """Mock for torch.nn.functional"""
            @staticmethod
            def silu(input, inplace=False):
                return input
                
            @staticmethod
            def linear(input, weight, bias=None):
                return input
                
            @staticmethod
            def softmax(input, dim=None, _stacklevel=3, dtype=None):
                return input
                
            @staticmethod
            def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
                return input
                
            @staticmethod
            def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
                        scale_grad_by_freq=False, sparse=False):
                return input

    class optim(ModuleMock):
        class AdamW(ModuleMock):
            def __init__(self, params, lr=1e-3, **kwargs):
                super().__init__()
                self.param_groups = [{'params': params}]
                self.defaults = {'lr': lr, **kwargs}
                
            def step(self, closure=None):
                pass
                
            def zero_grad(self, set_to_none=False):
                pass
                
            def state_dict(self):
                return {}
                
            def load_state_dict(self, state_dict):
                pass

    @contextlib.contextmanager
    def no_grad(self):
        yield
        
    def manual_seed(self, seed):
        pass
        
    def set_grad_enabled(self, mode):
        return self
        
    def is_grad_enabled(self):
        return False
        
    def cuda(self, *args, **kwargs):
        return self
        
    def float(self):
        return self
        
    def __getattr__(self, name):
        if name in ('__file__', '__path__'):
            return None
        return super().__getattr__(name)

    class Tensor(ModuleMock):
        pass
        
    class FloatTensor(ModuleMock):
        pass
        
    class device(ModuleMock):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.type = 'cuda' if args and args[0] == 'cuda' else 'cpu'
    
    @staticmethod
    def tensor(*args, **kwargs):
        return MockTorch.Tensor()
        
    @staticmethod
    def float32():
        return 'float32'

    @property
    def float16(self):
        return "float16"


# Mock transformers and its dependencies
class MockPreTrainedModel:
    def __init__(self, *args, **kwargs):
        pass
    
    def to(self, *args, **kwargs):
        return self
    
    def eval(self):
        return self
    
    def generate(self, *args, **kwargs):
        return torch.zeros(1, 10, dtype=torch.long)

class MockTokenizer:
    def __init__(self, *args, **kwargs):
        self.pad_token_id = 0
        self.eos_token_id = 1
    
    def __call__(self, *args, **kwargs):
        return {'input_ids': torch.zeros(1, 10, dtype=torch.long), 'attention_mask': torch.ones(1, 10, dtype=torch.long)}
    
    def encode(self, *args, **kwargs):
        return [0] * 10
    
    def decode(self, *args, **kwargs):
        return "Mock decoded text"

class Qwen3Config:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 32000
        self.hidden_size = 4096
        self.intermediate_size = 11008
        self.num_hidden_layers = 32
        self.num_attention_heads = 32
        self.num_key_value_heads = 32
        self.hidden_act = 'silu'
        self.max_position_embeddings = 32768
        self.initializer_range = 0.02
        self.rms_norm_eps = 1e-6
        self.use_cache = True
        self.tie_word_embeddings = False
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.torch_dtype = 'float16'
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class MockTransformers:
    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if 'qwen' in args[0].lower():
                return Qwen3Config()
            return {}

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return MockPreTrainedModel()
    
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return MockTokenizer()
    
    class PreTrainedModel:
        pass
    
    class PreTrainedTokenizer:
        pass
        
    # Add Qwen3 specific classes
    Qwen3Config = Qwen3Config

# Apply mocks
sys.modules["flash_attn"] = MockFlashAttn()
torch_mock = MockTorch()
sys.modules["torch"] = torch_mock
sys.modules["torch.cuda"] = torch_mock.cuda
sys.modules["torch.nn"] = torch_mock.nn
sys.modules["torch.optim"] = torch_mock.optim
sys.modules["transformers"] = MockTransformers()
sys.modules["transformers.utils"] = MagicMock()
sys.modules["transformers.utils.generic"] = MagicMock()
sys.modules["transformers.utils._pytree"] = MagicMock()

# Mock torch.utils._pytree
class MockPyTree:
    def tree_map(self, *args, **kwargs):
        return lambda x: x

if not hasattr(torch, 'utils'):
    torch.utils = MagicMock()
    torch.utils._pytree = MockPyTree()

import os
import tempfile
import pytest

@pytest.fixture
def temp_model_dir():
    """Create a temporary directory with a dummy config.json for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy config.json file
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            f.write('{"model_type":"bert","hidden_size":768,"num_hidden_layers":12}')
        yield tmpdir

# Skip tests that require CUDA if not available
HAS_CUDA = False


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (slow, requires external services)",
    )


# Skip tests that require CUDA if not available
def pytest_runtest_setup(item):
    """Skip tests that require CUDA if not available."""
    if not HAS_CUDA and any(item.iter_markers(name="cuda")):
        pytest.skip("Test requires CUDA")


# Add a command line option to run CUDA tests
def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption(
        "--run-cuda",
        action="store_true",
        default=False,
        help="run tests that require CUDA",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    skip_cuda = not config.getoption("--run-cuda")
    skip_slow = not config.getoption("--run-slow")
    skip_integration = not config.getoption("--run-integration")

    for item in items:
        if skip_cuda and "cuda" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="need --run-cuda option to run"))
        if skip_slow and "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="need --run-slow option to run"))
        if skip_integration and "integration" in item.keywords:
            item.add_marker(
                pytest.mark.skip(reason="need --run-integration option to run")
            )
