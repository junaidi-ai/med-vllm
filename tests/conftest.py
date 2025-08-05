"""Pytest configuration and fixtures for the test suite.

This module provides common test utilities, fixtures, and mock objects that can be
used across different test modules. It's designed to help with testing components
that depend on external libraries like transformers and torch.
"""
import os
import sys
import types
import importlib
import pkgutil
import contextlib
from enum import Enum
from typing import Any, Dict, List, Optional, Type, get_origin, get_args
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Ensure the medvllm package is properly importable
import medvllm

# Manually register the medvllm.medical.config package and its submodules
# This helps pytest discover the package structure correctly
medvllm_module_path = os.path.dirname(medvllm.__file__)
medical_config_path = os.path.join(medvllm_module_path, 'medical', 'config')

if os.path.exists(medical_config_path):
    # Import the package to ensure it's in sys.modules
    try:
        import medvllm.medical.config
        # Force import of submodules
        for _, module_name, _ in pkgutil.iter_modules([medical_config_path]):
            full_module_name = f'medvllm.medical.config.{module_name}'
            if full_module_name not in sys.modules:
                try:
                    importlib.import_module(full_module_name)
                except ImportError as e:
                    print(f"Warning: Could not import {full_module_name}: {e}")
    except ImportError as e:
        print(f"Warning: Could not import medvllm.medical.config: {e}")

def pytest_configure():
    """Configure pytest and set up mocks before test collection."""
    # Create a mock torch module
    torch = sys.modules['torch'] = types.ModuleType('torch')
    
    # Add nn module
    torch.nn = types.ModuleType('torch.nn')
    torch.nn.Module = type('Module', (), {
        '__init__': lambda self: None,
        'train': lambda self, mode=True: self,
        'eval': lambda self: self.train(False),
        'parameters': lambda self, recurse=True: [],
        'to': lambda self, *args, **kwargs: self,
        'state_dict': lambda self, *args, **kwargs: {},
        'load_state_dict': lambda self, state_dict, strict=True: None,
    })

    # Add tensor function and basic tensor operations
    def tensor(data, *args, **kwargs):
        if isinstance(data, (list, tuple)):
            return data[0] if data else None
        return data

    torch.tensor = tensor
    torch.Tensor = type('Tensor', (object,), {
        'to': lambda self, *args, **kwargs: self,
        'cuda': lambda self, *args, **kwargs: self,
        'cpu': lambda self: self,
        'shape': property(lambda self: (1,)),
        'device': 'cpu',
        'dtype': torch.float32,
    })

    # Add common nn modules
    torch.nn.Linear = type('Linear', (torch.nn.Module,), {
        '__init__': lambda self, in_features, out_features, bias=True: None,
        'forward': lambda self, x: x,
        'weight': torch.tensor([]),
        'bias': torch.tensor([]),
    })

    torch.nn.Embedding = type('Embedding', (torch.nn.Module,), {
        '__init__': lambda self, num_embeddings, embedding_dim, padding_idx=None: None,
        'forward': lambda self, x: x,
        'weight': torch.tensor([]),
    })

    torch.nn.Dropout = type('Dropout', (torch.nn.Module,), {
        '__init__': lambda self, p=0.5, inplace=False: None,
        'forward': lambda self, x: x,
    })

    # Add functional module
    torch.nn.functional = types.ModuleType('torch.nn.functional')
    torch.nn.functional.linear = lambda input, weight, bias=None: input
    torch.nn.functional.dropout = lambda input, p=0.5, training=True, inplace=False: input
    torch.nn.functional.softmax = lambda input, dim=None: input
    torch.nn.functional.cross_entropy = lambda input, target, *args, **kwargs: tensor(0.0)

    # Add common functions
    torch.cat = lambda tensors, dim=0: tensors[0] if tensors else None
    torch.stack = lambda tensors, dim=0: tensors[0] if tensors else None
    torch.tanh = lambda x: x
    torch.sigmoid = lambda x: x
    torch.softmax = lambda x, dim: x
    torch.matmul = lambda x, y: x * y  # Simple implementation for testing
    torch.no_grad = lambda: contextlib.nullcontext()

    # Add cuda module
    torch.cuda = types.ModuleType('torch.cuda')
    torch.cuda.is_available = lambda: False

    # Add backends module
    torch.backends = types.ModuleType('torch.backends')
    torch.backends.cuda = types.ModuleType('torch.backends.cuda')
    torch.backends.cuda.sdp_kernel = contextlib.nullcontext()

    # Add optim module
    torch.optim = types.ModuleType('torch.optim')
    torch.optim.Optimizer = type('Optimizer', (object,), {
        'zero_grad': lambda self: None,
        'step': lambda self: None,
        'state_dict': lambda self: {},
        'load_state_dict': lambda self, state_dict: None,
    })

    torch.optim.AdamW = type('AdamW', (torch.optim.Optimizer,), {
        '__init__': lambda self, params, **kwargs: None,
    })

    # Add autograd module
    torch.autograd = types.ModuleType('torch.autograd')
    torch.autograd.Variable = torch.Tensor
    torch.autograd.grad = lambda *args, **kwargs: (tensor(0.0),)
    torch.autograd.Function = type('Function', (object,), {
        'apply': lambda *args, **kwargs: tensor(0.0),
        'forward': lambda *args, **kwargs: (tensor(0.0),),
        'backward': lambda *args, **kwargs: (tensor(0.0),),
    })

    # Add distributed module
    torch.distributed = types.ModuleType('torch.distributed')
    torch.distributed.is_available = lambda: False
    torch.distributed.get_rank = lambda: 0
    torch.distributed.get_world_size = lambda: 1
    torch.distributed.is_initialized = lambda: False

    # Ensure the mock is used for all imports
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.nn.functional'] = torch.nn.functional
    sys.modules['torch.cuda'] = torch.cuda
    sys.modules['torch.backends'] = torch.backends
    sys.modules['torch.backends.cuda'] = torch.backends.cuda
    sys.modules['torch.optim'] = torch.optim
    sys.modules['torch.autograd'] = torch.autograd
    sys.modules['torch.distributed'] = torch.distributed

    # Now import the package to ensure it uses our mock
    import medvllm
    
    # Ensure the medvllm package is properly importable
    medvllm_module_path = os.path.dirname(medvllm.__file__)
    medical_config_path = os.path.join(medvllm_module_path, 'medical', 'config')

    if os.path.exists(medical_config_path):
        # Import the package to ensure it's in sys.modules
        try:
            import medvllm.medical.config
            # Force import of submodules
            for _, module_name, _ in pkgutil.iter_modules([medical_config_path]):
                full_module_name = f'medvllm.medical.config.{module_name}'
                if full_module_name not in sys.modules:
                    try:
                        importlib.import_module(full_module_name)
                    except ImportError as e:
                        print(f"Warning: Could not import {full_module_name}: {e}")
        except ImportError as e:
            print(f"Warning: Could not import medvllm.medical.config: {e}")

    yield  # Let the test run

    # Register the mock in sys.modules
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.nn.functional'] = torch.nn.functional
    sys.modules['torch.cuda'] = torch.cuda
    sys.modules['torch.backends'] = torch.backends
    sys.modules['torch.backends.cuda'] = torch.backends.cuda
    sys.modules['torch.optim'] = torch.optim
    sys.modules['torch.autograd'] = torch.autograd
    sys.modules['torch.distributed'] = torch.distributed

# Run the configuration
pytest_configure()

def create_module(name, **attrs):
    """Create a module with the given name and attributes."""
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module

# Create proper module hierarchy for torch
torch = create_module('torch')
torch.__version__ = '2.0.0'

# Add dtype attributes
torch.float16 = 'torch.float16'
torch.float32 = 'torch.float32'
torch.float64 = 'torch.float64'
torch.int32 = 'torch.int32'
torch.int64 = 'torch.int64'
torch.uint8 = 'torch.uint8'
torch.bool = 'torch.bool'

# Add dtype class
torch.dtype = type('dtype', (), {'__str__': lambda self: str(self)})()

# Add device class
torch.device = type('device', (), {
    '__init__': lambda self, device: None,
    'type': 'cpu',
    'index': 0,
    '__str__': lambda self: 'cpu'
})

# Add nn module
torch.nn = create_module('torch.nn')

# Add Module base class
torch.nn.Module = type('Module', (object,), {
    '__init__': lambda self: None,
    'train': lambda self, mode=True: self,
    'eval': lambda self: self.train(False),
    'parameters': lambda self, recurse=True: [],
    'to': lambda self, *args, **kwargs: self,
    'state_dict': lambda self, *args, **kwargs: {},
    'load_state_dict': lambda self, state_dict, strict=True: None,
})

# Add tensor function and basic tensor operations
def tensor(data, *args, **kwargs):
    if isinstance(data, (list, tuple)):
        return data[0] if data else None
    return data

torch.tensor = tensor
torch.Tensor = type('Tensor', (object,), {
    'to': lambda self, *args, **kwargs: self,
    'cuda': lambda self, *args, **kwargs: self,
    'cpu': lambda self: self,
    'shape': property(lambda self: (1,)),
    'device': 'cpu',
    'dtype': torch.float32,
})

# Add common nn modules
torch.nn.Linear = type('Linear', (torch.nn.Module,), {
    '__init__': lambda self, in_features, out_features, bias=True: None,
    'forward': lambda self, x: x,
    'weight': torch.tensor([]),
    'bias': torch.tensor([]),
})

torch.nn.Embedding = type('Embedding', (torch.nn.Module,), {
    '__init__': lambda self, num_embeddings, embedding_dim, padding_idx=None: None,
    'forward': lambda self, x: x,
    'weight': torch.tensor([]),
})

torch.nn.Dropout = type('Dropout', (torch.nn.Module,), {
    '__init__': lambda self, p=0.5, inplace=False: None,
    'forward': lambda self, x: x,
})

# Add functional module
torch.nn.functional = create_module('torch.nn.functional')
torch.nn.functional.linear = lambda input, weight, bias=None: input

# Add common functions
torch.cat = lambda tensors, dim=0: tensors[0] if tensors else None
torch.stack = lambda tensors, dim=0: tensors[0] if tensors else None
torch.tanh = lambda x: x
torch.sigmoid = lambda x: x
torch.softmax = lambda x, dim: x
torch.matmul = lambda x, y: x * y  # Simple implementation for testing
torch.no_grad = lambda: contextlib.nullcontext()

torch.cuda = create_module('torch.cuda')
torch.cuda.is_available = lambda: False

torch.backends = create_module('torch.backends')
torch.backends.cuda = create_module('torch.backends.cuda')
torch.backends.cuda.sdp_kernel = contextlib.nullcontext()

# Add optim module
torch.optim = create_module('torch.optim')
torch.optim.Optimizer = type('Optimizer', (object,), {
    'zero_grad': lambda self: None,
    'step': lambda self: None,
    'state_dict': lambda self: {},
    'load_state_dict': lambda self, state_dict: None,
})

torch.optim.AdamW = type('AdamW', (torch.optim.Optimizer,), {
    '__init__': lambda self, params, **kwargs: None,
})

# Add nn.functional module
torch.nn.functional = create_module('torch.nn.functional')
torch.nn.functional.linear = lambda input, weight, bias=None: input
torch.nn.functional.dropout = lambda input, p=0.5, training=True, inplace=False: input
torch.nn.functional.softmax = lambda input, dim=None: input
torch.nn.functional.cross_entropy = lambda input, target, *args, **kwargs: tensor(0.0)

# Add autograd module
torch.autograd = create_module('torch.autograd')
torch.autograd.Variable = torch.Tensor

torch.autograd.grad = lambda *args, **kwargs: (tensor(0.0),)

torch.autograd.Function = type('Function', (object,), {
    'apply': lambda *args, **kwargs: tensor(0.0),
    'forward': lambda *args, **kwargs: (tensor(0.0),),
    'backward': lambda *args, **kwargs: (tensor(0.0),),
})

# Add distributed module
torch.distributed = create_module('torch.distributed')
torch.distributed.is_available = lambda: False

torch.distributed.get_rank = lambda: 0
torch.distributed.get_world_size = lambda: 1
torch.distributed.is_initialized = lambda: False

# Add functional API
torch.nn.functional = create_module('torch.nn.functional')
torch.nn.functional.linear = lambda input, weight, bias=None: input
torch.nn.functional.softmax = lambda input, dim=None, _stacklevel=3, dtype=None: input
torch.nn.functional.dropout = lambda input, p=0.5, training=True, inplace=False: input

# Add tensor types
torch.Tensor = type('Tensor', (object,), {
    '__init__': lambda self, *args, **kwargs: None,
    'to': lambda self, *args, **kwargs: self,
    'cuda': lambda self, *args, **kwargs: self,
    'cpu': lambda self: self,
    'numpy': lambda self: None,
    'shape': (1,),
    'dtype': torch.float32,
    'device': 'cpu',
})

torch.FloatTensor = type('FloatTensor', (torch.Tensor,), {})
torch.LongTensor = type('LongTensor', (torch.Tensor,), {})
torch.IntTensor = type('IntTensor', (torch.Tensor,), {})
torch.BoolTensor = type('BoolTensor', (torch.Tensor,), {})

# Add device support
torch.device = lambda device: device
torch.cuda = create_module('torch.cuda')
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

# Add tensor creation functions
torch.tensor = lambda x, *args, **kwargs: x
torch.zeros = lambda *args, **kwargs: None
torch.ones = lambda *args, **kwargs: None
torch.randn = lambda *args, **kwargs: None
torch.arange = lambda *args, **kwargs: None
torch.cat = lambda tensors, dim=0: tensors[0] if tensors else None
torch.stack = lambda tensors, dim=0: tensors[0] if tensors else None

# Add nn module
torch.nn = create_module('torch.nn')
torch.nn.Module = type('Module', (), {'__init__': lambda self: None})
torch.nn.Embedding = type('Embedding', (torch.nn.Module,), {
    '__init__': lambda self, num_embeddings, embedding_dim, padding_idx=None, **kwargs: None,
    'forward': lambda self, x: x  # Identity function for forward pass
})

torch.nn.Linear = type('Linear', (torch.nn.Module,), {
    '__init__': lambda self, in_features, out_features, bias=True: None,
    'forward': lambda self, x: x  # Identity function for forward pass
})

# Add basic tensor types
torch.Tensor = type('Tensor', (), {'__init__': lambda *args, **kwargs: None})
torch.is_available = lambda: True
torch.device_count = lambda: 1

# Add dtypes
torch.float16 = type('dtype', (), {'__str__': lambda: 'torch.float16'})()
torch.float32 = type('dtype', (), {'__str__': lambda: 'torch.float32'})()
torch.int64 = type('dtype', (), {'__str__': lambda: 'torch.int64'})()
torch.dtype = type('dtype', (), {'__str__': lambda: 'torch.dtype'})

# Add tensor types
torch.LongTensor = type('LongTensor', (torch.Tensor,), {'dtype': torch.int64})
torch.FloatTensor = type('FloatTensor', (torch.Tensor,), {'dtype': torch.float32})
torch.HalfTensor = type('HalfTensor', (torch.Tensor,), {'dtype': torch.float16})
torch.BoolTensor = type('BoolTensor', (torch.Tensor,), {'dtype': bool})

# Add tensor operations
def tensor(data, *args, **kwargs):
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], bool):
        return torch.BoolTensor(data, *args, **kwargs)
    elif isinstance(data, (list, tuple)) and data and isinstance(data[0], (int, float)):
        return torch.FloatTensor(data, *args, **kwargs)
    return torch.Tensor(data, *args, **kwargs)

torch.tensor = tensor

# Add common mathematical functions
torch.tanh = lambda x: x  # Simple identity function for tanh
torch.sigmoid = lambda x: 1 / (1 + (-x).exp()) if hasattr(x, 'exp') else 1 / (1 + (-x).exp())
torch.exp = lambda x: x.exp() if hasattr(x, 'exp') else x  # Simple mock for exp
torch.cat = lambda tensors, dim=0: tensors[0].__class__(sum((t.tolist() for t in tensors), []))
torch.stack = lambda tensors, dim=0: tensors[0].__class__([t.tolist() for t in tensors])

# Add nn module
torch.nn = types.ModuleType('torch.nn')
torch.nn.Module = type('Module', (), {'__init__': lambda self: None})

# Common nn modules
torch.nn.Linear = type('Linear', (torch.nn.Module,), {
    '__init__': lambda self, in_features, out_features, bias=True: None,
    'forward': lambda self, x: x
})

torch.nn.ReLU = type('ReLU', (torch.nn.Module,), {
    '__init__': lambda self, inplace=False: None,
    'forward': lambda self, x: x if hasattr(x, '__ge__') and x >= 0 else x
})

torch.nn.Embedding = type('Embedding', (torch.nn.Module,), {
    '__init__': lambda self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, 
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, 
                 _freeze=False, device=None, dtype=None: None,
    'forward': lambda self, x: x
})

torch.nn.Dropout = type('Dropout', (torch.nn.Module,), {
    '__init__': lambda self, p=0.5, inplace=False: None,
    'forward': lambda self, x: x
})

torch.nn.LayerNorm = type('LayerNorm', (torch.nn.Module,), {
    '__init__': lambda self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None: None,
    'forward': lambda self, x: x
})

# Add functional module
torch.nn.functional = types.ModuleType('torch.nn.functional')
torch.nn.functional.relu = lambda x, inplace=False: x if hasattr(x, '__ge__') and x >= 0 else x
torch.nn.functional.dropout = lambda x, p=0.5, training=True, inplace=False: x
torch.nn.functional.softmax = lambda x, dim=None, _stacklevel=3, dtype=None: x
torch.nn.functional.gelu = lambda x: x

# Add autograd module
torch.autograd = types.ModuleType('torch.autograd')
torch.autograd.Variable = torch.Tensor  # Simple mock for Variable

torch.no_grad = lambda: contextlib.nullcontext()  # Simple mock for no_grad

# Add device class
torch.device = type('device', (), {
    '__init__': lambda self, device, *args, **kwargs: setattr(self, 'type', device),
    '__str__': lambda self: f"{self.type}",
    '__repr__': lambda self: f"device(type='{self.type}')"
})

torch.cuda = create_module('torch.cuda')
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

torch.cuda.Stream = type('Stream', (), {
    '__init__': lambda *args, **kwargs: None,
    '__enter__': lambda self: self,
    '__exit__': lambda *args: None,
    'synchronize': lambda: None
})
torch.cuda.current_stream = lambda device=None: torch.cuda.Stream()

torch.nn = create_module('torch.nn')
torch.nn.Module = type('Module', (), {'__init__': lambda *args, **kwargs: None})
torch.nn.Linear = type('Linear', (), {'__init__': lambda *args, **kwargs: None})

torch.nn.functional = create_module('torch.nn.functional')
torch.nn.functional.linear = lambda *args, **kwargs: torch.Tensor()
torch.nn.functional.relu = lambda *args, **kwargs: torch.Tensor()
torch.nn.functional.dropout = lambda *args, **kwargs: args[0] if args else torch.Tensor()
torch.nn.functional.layer_norm = lambda *args, **kwargs: args[0] if args else torch.Tensor()

torch.distributed = create_module('torch.distributed')
torch.distributed.get_rank = lambda: 0

# Create and register multiprocessing module
mp_module = create_module('torch.multiprocessing')
mp_module.Process = type('Process', (), {
    '__init__': lambda self, *args, **kwargs: None,
    'start': lambda self: None,
    'join': lambda self, timeout=None: None,
    'is_alive': lambda self: False,
    'terminate': lambda self: None
})
mp_module.Queue = type('Queue', (), {
    '__init__': lambda self, *args, **kwargs: None,
    'put': lambda self, item, *args, **kwargs: None,
    'get': lambda self, *args, **kwargs: None,
    'empty': lambda self: True
})

# Add torch.multiprocessing to sys.modules
torch.multiprocessing = mp_module
sys.modules['torch.multiprocessing'] = mp_module

# Now set up the rest of the mocks
import sys
if 'transformers' not in sys.modules:
    transformers = create_module('transformers')
    sys.modules['transformers'] = transformers
else:
    transformers = sys.modules['transformers']

transformers.__file__ = '/mock/transformers/__init__.py'
transformers.__version__ = '4.30.0'

# Import pydantic and its components
import pydantic
from pydantic import BaseModel, ValidationError, Field

# Add common classes and functions
transformers.PreTrainedModel = type('PreTrainedModel', (object,), {
    '__init__': lambda self, *args, **kwargs: None,
    'config': {},
    'eval': lambda self: self,
    'train': lambda self, mode=True: self,
    'parameters': lambda self: [],
    'to': lambda self, *args, **kwargs: self,
    'save_pretrained': lambda self, *args, **kwargs: None,
})

transformers.PretrainedConfig = type('PretrainedConfig', (object,), {
    '__init__': lambda self, *args, **kwargs: None,
    'to_dict': lambda self: {},
    'from_pretrained': classmethod(lambda cls, *args, **kwargs: cls()),
})

transformers.PreTrainedTokenizerBase = type('PreTrainedTokenizerBase', (object,), {
    '__init__': lambda self, *args, **kwargs: None,
    'pad_token_id': 0,
    'eos_token_id': 1,
    'bos_token_id': 2,
    'unk_token_id': 3,
    'pad_token': '[PAD]',
    'eos_token': '</s>',
    'bos_token': '<s>',
    'unk_token': '<unk>',
    'name_or_path': 'mock-model',
    'vocab_size': 30000,
    'model_max_length': 512,
    'is_fast': False,
    'padding_side': 'right',
    'truncation_side': 'right',
    'return_tensors': 'pt',
    'pad': lambda self, *args, **kwargs: {'input_ids': [], 'attention_mask': []},
    'encode_plus': lambda self, *args, **kwargs: {'input_ids': [], 'attention_mask': []},
    'decode': lambda self, *args, **kwargs: 'decoded text',
    'batch_decode': lambda self, *args, **kwargs: ['decoded text'],
    'save_pretrained': lambda self, *args, **kwargs: None,
})

# Add AutoModel classes
transformers.AutoModel = type('AutoModel', (object,), {
    'from_pretrained': classmethod(lambda cls, *args, **kwargs: transformers.PreTrainedModel()),
})

transformers.AutoModelForSequenceClassification = type('AutoModelForSequenceClassification', (object,), {
    'from_pretrained': classmethod(lambda cls, *args, **kwargs: transformers.PreTrainedModel()),
})

transformers.AutoModelForTokenClassification = type('AutoModelForTokenClassification', (object,), {
    'from_pretrained': classmethod(lambda cls, *args, **kwargs: transformers.PreTrainedModel()),
})

transformers.AutoTokenizer = type('AutoTokenizer', (object,), {
    'from_pretrained': classmethod(lambda cls, *args, **kwargs: transformers.PreTrainedTokenizerBase()),
    'eos_token_id': 1,
    'bos_token_id': 2,
    'pad_token_id': 0,
    'unk_token_id': 3,
    'model_max_length': 512,
    'is_fast': False,
    'padding_side': 'right',
    'truncation_side': 'right',
    '__call__': lambda *args, **kwargs: {'input_ids': [[1, 2, 3]], 'attention_mask': [[1, 1, 1]]}
})

transformers.AutoConfig = type('AutoConfig', (object,), {
    'from_pretrained': classmethod(lambda cls, *args, **kwargs: transformers.PretrainedConfig()),
    'eos_token_id': 1,
    'bos_token_id': 2,
    'pad_token_id': 0,
    'unk_token_id': 3,
    'model_max_length': 512,
    'is_fast': False,
    'padding_side': 'right',
    'truncation_side': 'right',
    '__call__': lambda *args, **kwargs: {'input_ids': [[1, 2, 3]], 'attention_mask': [[1, 1, 1]]}
})

transformers.PreTrainedModel = type('PreTrainedModel', (), {
    'config_class': type('MockConfig', (), {
        '__init__': lambda self, model_type="qwen3", **kwargs: (
            setattr(self, 'model_type', model_type) or
            setattr(self, 'vocab_size', 32000) or
            setattr(self, 'hidden_size', 4096) or
            setattr(self, 'num_hidden_layers', 32) or
            setattr(self, 'num_attention_heads', 32) or
            [setattr(self, k, v) for k, v in kwargs.items()]
        ),
        'to_dict': lambda self: {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    }),
    'from_pretrained': classmethod(lambda cls, *args, **kwargs: cls()),
    '__init__': lambda self, config=None: setattr(self, 'config', config or transformers.PreTrainedModel.config_class())
})

# Add submodules
transformers.models = create_module('transformers.models')
transformers.models.auto = create_module('transformers.models.auto')
transformers.models.auto.configuration_auto = create_module('transformers.models.auto.configuration_auto')

transformers.utils = create_module('transformers.utils')
transformers.utils.versions = type('Versions', (), {
    'require_version': lambda *args, **kwargs: None
})()
transformers.utils.logging = create_module('transformers.utils.logging')
transformers.utils.logging.get_logger = lambda name: MagicMock(
    debug=print, info=print, warning=print, error=print
)

transformers.tokenization_utils_base = create_module('transformers.tokenization_utils_base')
transformers.tokenization_utils_base.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase

transformers.configuration_utils = create_module('transformers.configuration_utils')
transformers.configuration_utils.PretrainedConfig = transformers.PreTrainedModel.config_class

transformers.modeling_utils = create_module('transformers.modeling_utils')
transformers.modeling_utils.PreTrainedModel = transformers.PreTrainedModel

# Add tree_map function
transformers.tree_map = lambda fn, *args, **kwargs: fn(*args) if args else None

# Add ModelOutput class
transformers.utils.ModelOutput = type('ModelOutput', (dict,), {
    '__getattr__': lambda self, k: self[k],
    '__setattr__': lambda self, k, v: self.__setitem__(k, v)
})

# pydantic is already imported at the top of the file

# Add to sys.modules
sys.modules.update({
    'torch': torch,
    'torch.nn': torch.nn,
    'torch.nn.functional': torch.nn.functional,
    'torch.cuda': torch.cuda,
    'torch.distributed': torch.distributed,
    'transformers': transformers,
    'transformers.models': transformers.models,
    'transformers.models.auto': transformers.models.auto,
    'transformers.models.auto.configuration_auto': transformers.models.auto.configuration_auto,
    'transformers.utils': transformers.utils,
    'transformers.utils.versions': transformers.utils.versions,
    'transformers.utils.logging': transformers.utils.logging,
    'transformers.tokenization_utils_base': transformers.tokenization_utils_base,
    'transformers.configuration_utils': transformers.configuration_utils,
    'transformers.modeling_utils': transformers.modeling_utils,
    'pydantic': pydantic,
    'pydantic.BaseModel': BaseModel,
    'pydantic.ValidationError': ValidationError,
    'pydantic.Field': Field
})

# Fixtures
@pytest.fixture
def mock_transformers():
    """Fixture that provides access to the mocked transformers module."""
    return transformers

@pytest.fixture
def mock_torch():
    """Fixture that provides access to the mocked torch module."""
    return torch

@pytest.fixture
def mock_model_config():
    """Fixture that provides a mock model configuration."""
    return transformers.PreTrainedModel.config_class()

@pytest.fixture
def mock_tokenizer():
    """Fixture that provides a mock tokenizer."""
    return transformers.PreTrainedTokenizerBase()
