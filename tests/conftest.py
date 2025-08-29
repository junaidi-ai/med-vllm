"""Pytest configuration and fixtures for the test suite.

This module provides common test utilities, fixtures, and mock objects that can be
used across different test modules. It's designed to help with testing components
that depend on external libraries like transformers and torch.
"""

import sys
import os
import pytest
import types
import contextlib
import warnings
import importlib.util
import pkgutil
from unittest.mock import MagicMock

# Suppress noisy Pydantic "protected namespace" warnings in tests
warnings.filterwarnings(
    "ignore",
    message=r".*protected namespace.*",
    category=UserWarning,
    module=r"pydantic\..*",
)

# Add the project root to the Python path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# NOTE: Do not import real torch here; we provide a stub/mock for tests

# Import our mock adapters and models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the patching functions from our separate module
from tests.conftest_patches import (
    patch_adapters,
    patch_medical_model,
    patch_transformers,
    patch_medical_config,
    patch_datasets,
)

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Defer importing medvllm until after mocks are applied in pytest_configure


def pytest_configure(config):
    """Configure pytest and set up mocks before test collection.

    - Ensure mocks/patches are applied once at session start
    - Defer importing medvllm until after mocks are ready
    """
    # Apply transformers/adapters/models/config patches
    try:
        patch_transformers()
        # Defensive: ensure AutoTokenizer exists on the mocked transformers
        try:
            import sys as _sys

            _t = _sys.modules.get("transformers")
            if _t is not None and not hasattr(_t, "AutoTokenizer"):

                class _AutoTok:
                    @classmethod
                    def from_pretrained(cls, *args, **kwargs):
                        inst = cls()
                        setattr(inst, "pad_token_id", 0)
                        return inst

                setattr(_t, "AutoTokenizer", _AutoTok)
        except Exception:
            pass
        patch_adapters()
        patch_medical_model()
        patch_medical_config()
        patch_datasets()
    except Exception as e:
        print(f"Warning applying test patches: {e}")

    # Import medvllm and force-load medical.config submodules for discovery
    try:
        import medvllm

        medvllm_module_path = os.path.dirname(medvllm.__file__)
        medical_config_path = os.path.join(medvllm_module_path, "medical", "config")
        if os.path.exists(medical_config_path):
            try:
                import medvllm.medical.config

                for _, module_name, _ in pkgutil.iter_modules([medical_config_path]):
                    full_module_name = f"medvllm.medical.config.{module_name}"
                    if full_module_name not in sys.modules:
                        try:
                            importlib.import_module(full_module_name)
                        except ImportError as e:
                            print(f"Warning: Could not import {full_module_name}: {e}")
            except ImportError as e:
                print(f"Warning: Could not import medvllm.medical.config: {e}")
    except ImportError as e:
        print(f"Warning: Could not import medvllm: {e}")


# Note: Do not call pytest_configure() manually; pytest invokes it automatically.


def create_module(name, **attrs):
    """Create a module with the given name and attributes."""
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


class MockTensor:
    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        self.data = data
        self.dtype = dtype or type("dtype", (), {"__name__": "torch.float32"})()
        self.device = device or "cpu"
        self.requires_grad = requires_grad
        self.shape = (len(data),) if hasattr(data, "__len__") else (1,)

    def to(self, *args, **kwargs):
        if args and (
            args[0] in (torch.float32, torch.float64, torch.long, torch.int64)
            or isinstance(args[0], str)
            and args[0] in ("cpu", "cuda")
        ):
            return self
        return self

    def cuda(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np

        return np.array(self.data)

    def __len__(self):
        return len(self.data) if hasattr(self.data, "__len__") else 1

    def __getitem__(self, idx):
        return self.data[idx] if hasattr(self.data, "__getitem__") else self.data

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim] if dim < len(self.shape) else 1

    def view(self, *args):
        return self

    def __str__(self):
        return f"MockTensor({self.data}, dtype={self.dtype}, device={self.device})"

    def __repr__(self):
        return self.__str__()


# Create proper module hierarchy for torch
torch = create_module("torch")
torch.__version__ = "2.0.0"

# Add common dtypes
torch.float32 = type("dtype", (), {"__name__": "torch.float32"})()
torch.float64 = type("dtype", (), {"__name__": "torch.float64"})()
torch.long = type("dtype", (), {"__name__": "torch.long"})()
torch.int64 = type("dtype", (), {"__name__": "torch.int64"})()

# Add device handling
torch.device = lambda device: f"device({device})"

# Add cuda module
torch.cuda = create_module("torch.cuda")
torch.cuda.is_available = lambda: False


# Provide a no-op compile decorator/function
def _noop_compile(fn=None, *args, **kwargs):
    if fn is None:

        def decorator(f):
            return f

        return decorator
    return fn


torch.compile = _noop_compile

# Add nn module
torch.nn = create_module("torch.nn")
torch.nn.functional = create_module("torch.nn.functional")
torch.optim = create_module("torch.optim")


# Set up tensor creation functions
def tensor(data, *args, **kwargs):
    return MockTensor(data, **kwargs)


def ones(*size, **kwargs):
    return MockTensor([1] * (size[0] if size else 1), **kwargs)


def zeros(*size, **kwargs):
    return MockTensor([0] * (size[0] if size else 1), **kwargs)


def randn(*size, **kwargs):
    import random

    data = [random.random() for _ in range(size[0] if size else 1)]
    return MockTensor(data, **kwargs)


def arange(*args, **kwargs):
    start = 0
    end = args[0]
    if len(args) > 1:
        start = args[0]
        end = args[1]
    return MockTensor(list(range(start, end)), **kwargs)


torch.tensor = tensor
torch.ones = ones
torch.zeros = zeros
torch.randn = randn
torch.arange = arange

# Set up Tensor class
torch.Tensor = MockTensor
torch.utils = create_module("torch.utils")
torch.utils.data = create_module("torch.utils.data")
torch.distributed = create_module("torch.distributed")
torch.autograd = create_module("torch.autograd")

# Mock CUDA functions
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.current_device = lambda: 0
torch.cuda.set_device = lambda *args, **kwargs: None

# Mock tensor creation functions
torch.tensor = lambda *args, **kwargs: args[0] if args else None
torch.Tensor = type(
    "Tensor",
    (object,),
    {
        "to": lambda self, *args, **kwargs: self,
        "cuda": lambda self, *args, **kwargs: self,
        "cpu": lambda self: self,
        "numpy": lambda self: None,
        "detach": lambda self: self,
        "requires_grad_": lambda self, requires_grad=True: self,
        "shape": (1,),
        "dtype": torch.float32,
        "device": "cpu",
        "__array_interface__": {"shape": (1,), "typestr": "<f4"},
    },
)

torch.FloatTensor = torch.Tensor
torch.LongTensor = torch.Tensor
torch.IntTensor = torch.Tensor

# Mock nn.Module
torch.nn.Module = type(
    "Module",
    (object,),
    {
        "__init__": lambda self: None,
        "parameters": lambda self: [],
        "to": lambda self, *args, **kwargs: self,
        "eval": lambda self: self,
        "train": lambda self, mode=True: self,
        "state_dict": lambda self: {},
        "load_state_dict": lambda self, state_dict, strict=True: None,
        "forward": lambda self, *args, **kwargs: None,
        "cuda": lambda self, device=None: self,
        "cpu": lambda self: self,
        "zero_grad": lambda self: None,
        "requires_grad_": lambda self, requires_grad=True: self,
    },
)

# Mock nn.functional
torch.nn.functional.softmax = lambda x, dim=-1: x
torch.nn.functional.log_softmax = lambda x, dim=-1: x
torch.nn.functional.cross_entropy = lambda input, target, *args, **kwargs: torch.tensor(0.0)
torch.nn.functional.mse_loss = lambda input, target, *args, **kwargs: torch.tensor(0.0)

# Mock optim
torch.optim.Adam = type(
    "Adam",
    (object,),
    {
        "__init__": lambda self, params, **kwargs: None,
        "step": lambda self, closure=None: None,
        "zero_grad": lambda self, set_to_none=False: None,
        "state_dict": lambda self: {},
        "load_state_dict": lambda self, state_dict: None,
    },
)

torch.optim.AdamW = torch.optim.Adam

# Mock autograd
torch.no_grad = contextlib.nullcontext
torch.enable_grad = contextlib.nullcontext
torch.set_grad_enabled = lambda mode: contextlib.nullcontext()

torch.autograd.Variable = torch.Tensor
torch.autograd.Function = type(
    "Function",
    (object,),
    {
        "forward": lambda *args, **kwargs: args[0] if args else None,
        "backward": lambda *args, **kwargs: None,
    },
)

# Mock distributed
torch.distributed.is_available = lambda: False
torch.distributed.is_initialized = lambda: False

# Mock utils
torch.utils.data.DataLoader = type(
    "DataLoader",
    (object,),
    {
        "__init__": lambda self, dataset=None, batch_size=1, shuffle=False, **kwargs: None,
        "__iter__": lambda self: iter([]),
        "__len__": lambda self: 0,
    },
)

# Mock random functions
torch.randn = lambda *args, **kwargs: torch.Tensor()
torch.rand = lambda *args, **kwargs: torch.Tensor()
torch.zeros = lambda *args, **kwargs: torch.Tensor()
torch.ones = lambda *args, **kwargs: torch.Tensor()
torch.empty = lambda *args, **kwargs: torch.Tensor()
torch.arange = lambda *args, **kwargs: torch.Tensor()
torch.linspace = lambda *args, **kwargs: torch.Tensor()

torch.tensor = lambda *args, **kwargs: args[0] if args else None

# Add dtype attributes
torch.float16 = "torch.float16"
torch.float32 = "torch.float32"
torch.float64 = "torch.float64"
torch.int32 = "torch.int32"
torch.int64 = "torch.int64"
torch.uint8 = "torch.uint8"
torch.bool = "torch.bool"

# Add dtype class
torch.dtype = type("dtype", (), {"__str__": lambda self: str(self)})()

# Add device class
torch.device = type(
    "device",
    (),
    {
        "__init__": lambda self, device: None,
        "type": "cpu",
        "index": 0,
        "__str__": lambda self: "cpu",
    },
)

# Add nn module
torch.nn = create_module("torch.nn")

# Add Module base class
torch.nn.Module = type(
    "Module",
    (object,),
    {
        "__init__": lambda self: None,
        "train": lambda self, mode=True: self,
        "eval": lambda self: self.train(False),
        "parameters": lambda self, recurse=True: [],
        "to": lambda self, *args, **kwargs: self,
        "state_dict": lambda self, *args, **kwargs: {},
        "load_state_dict": lambda self, state_dict, strict=True: None,
    },
)


# Add tensor function and basic tensor operations
def tensor(data, *args, **kwargs):
    if isinstance(data, (list, tuple)):
        return data[0] if data else None
    return data


torch.tensor = tensor
torch.Tensor = type(
    "Tensor",
    (object,),
    {
        "to": lambda self, *args, **kwargs: self,
        "cuda": lambda self, *args, **kwargs: self,
        "cpu": lambda self: self,
        "shape": property(lambda self: (1,)),
        "device": "cpu",
        "dtype": torch.float32,
    },
)

# Add common nn modules
torch.nn.Linear = type(
    "Linear",
    (torch.nn.Module,),
    {
        "__init__": lambda self, in_features, out_features, bias=True: None,
        "forward": lambda self, x: x,
        "weight": torch.tensor([]),
        "bias": torch.tensor([]),
    },
)

torch.nn.Embedding = type(
    "Embedding",
    (torch.nn.Module,),
    {
        "__init__": lambda self, num_embeddings, embedding_dim, padding_idx=None: None,
        "forward": lambda self, x: x,
        "weight": torch.tensor([]),
    },
)

torch.nn.Dropout = type(
    "Dropout",
    (torch.nn.Module,),
    {
        "__init__": lambda self, p=0.5, inplace=False: None,
        "forward": lambda self, x: x,
    },
)

# Add functional module
torch.nn.functional = create_module("torch.nn.functional")
torch.nn.functional.linear = lambda input, weight, bias=None: input

# Add common functions
torch.cat = lambda tensors, dim=0: tensors[0] if tensors else None
torch.stack = lambda tensors, dim=0: tensors[0] if tensors else None
torch.tanh = lambda x: x
torch.sigmoid = lambda x: x
torch.softmax = lambda x, dim: x
torch.matmul = lambda x, y: x * y  # Simple implementation for testing
torch.no_grad = lambda: contextlib.nullcontext()

torch.cuda = create_module("torch.cuda")
torch.cuda.is_available = lambda: False

torch.backends = create_module("torch.backends")
torch.backends.cuda = create_module("torch.backends.cuda")
torch.backends.cuda.sdp_kernel = contextlib.nullcontext()

# Add optim module
torch.optim = create_module("torch.optim")
torch.optim.Optimizer = type(
    "Optimizer",
    (object,),
    {
        "zero_grad": lambda self: None,
        "step": lambda self: None,
        "state_dict": lambda self: {},
        "load_state_dict": lambda self, state_dict: None,
    },
)

torch.optim.AdamW = type(
    "AdamW",
    (torch.optim.Optimizer,),
    {
        "__init__": lambda self, params, **kwargs: None,
    },
)

# Add nn.functional module
torch.nn.functional = create_module("torch.nn.functional")
torch.nn.functional.linear = lambda input, weight, bias=None: input
torch.nn.functional.dropout = lambda input, p=0.5, training=True, inplace=False: input
torch.nn.functional.softmax = lambda input, dim=None: input
torch.nn.functional.cross_entropy = lambda input, target, *args, **kwargs: tensor(0.0)

# Add autograd module
torch.autograd = create_module("torch.autograd")
torch.autograd.Variable = torch.Tensor

torch.autograd.grad = lambda *args, **kwargs: (tensor(0.0),)

torch.autograd.Function = type(
    "Function",
    (object,),
    {
        "apply": lambda *args, **kwargs: tensor(0.0),
        "forward": lambda *args, **kwargs: (tensor(0.0),),
        "backward": lambda *args, **kwargs: (tensor(0.0),),
    },
)

# Add distributed module
torch.distributed = create_module("torch.distributed")
torch.distributed.is_available = lambda: False

torch.distributed.get_rank = lambda: 0
torch.distributed.get_world_size = lambda: 1
torch.distributed.is_initialized = lambda: False

# Add functional API
torch.nn.functional = create_module("torch.nn.functional")
torch.nn.functional.linear = lambda input, weight, bias=None: input
torch.nn.functional.softmax = lambda input, dim=None, _stacklevel=3, dtype=None: input
torch.nn.functional.dropout = lambda input, p=0.5, training=True, inplace=False: input

# Add tensor types
torch.Tensor = type(
    "Tensor",
    (object,),
    {
        "__init__": lambda self, *args, **kwargs: None,
        "to": lambda self, *args, **kwargs: self,
        "cuda": lambda self, *args, **kwargs: self,
        "cpu": lambda self: self,
        "numpy": lambda self: None,
        "shape": (1,),
        "dtype": torch.float32,
        "device": "cpu",
    },
)

torch.FloatTensor = type("FloatTensor", (torch.Tensor,), {})
torch.LongTensor = type("LongTensor", (torch.Tensor,), {})
torch.IntTensor = type("IntTensor", (torch.Tensor,), {})
torch.BoolTensor = type("BoolTensor", (torch.Tensor,), {})

# Add device support
torch.device = lambda device: device
torch.cuda = create_module("torch.cuda")
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
torch.nn = create_module("torch.nn")
torch.nn.Module = type("Module", (), {"__init__": lambda self: None})
torch.nn.Embedding = type(
    "Embedding",
    (torch.nn.Module,),
    {
        "__init__": lambda self, num_embeddings, embedding_dim, padding_idx=None, **kwargs: None,
        "forward": lambda self, x: x,  # Identity function for forward pass
    },
)

torch.nn.Linear = type(
    "Linear",
    (torch.nn.Module,),
    {
        "__init__": lambda self, in_features, out_features, bias=True: None,
        "forward": lambda self, x: x,  # Identity function for forward pass
    },
)

# Add basic tensor types
torch.Tensor = type("Tensor", (), {"__init__": lambda *args, **kwargs: None})
torch.is_available = lambda: True
torch.device_count = lambda: 1

# Add dtypes
torch.float16 = type("dtype", (), {"__str__": lambda: "torch.float16"})()
torch.float32 = type("dtype", (), {"__str__": lambda: "torch.float32"})()
torch.int64 = type("dtype", (), {"__str__": lambda: "torch.int64"})()
torch.dtype = type("dtype", (), {"__str__": lambda: "torch.dtype"})

# Add tensor types
torch.LongTensor = type("LongTensor", (torch.Tensor,), {"dtype": torch.int64})
torch.FloatTensor = type("FloatTensor", (torch.Tensor,), {"dtype": torch.float32})
torch.HalfTensor = type("HalfTensor", (torch.Tensor,), {"dtype": torch.float16})
torch.BoolTensor = type("BoolTensor", (torch.Tensor,), {"dtype": bool})


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
torch.sigmoid = lambda x: 1 / (1 + (-x).exp()) if hasattr(x, "exp") else 1 / (1 + (-x).exp())
torch.exp = lambda x: x.exp() if hasattr(x, "exp") else x  # Simple mock for exp
torch.cat = lambda tensors, dim=0: tensors[0].__class__(sum((t.tolist() for t in tensors), []))
torch.stack = lambda tensors, dim=0: tensors[0].__class__([t.tolist() for t in tensors])

# Add nn module
torch.nn = types.ModuleType("torch.nn")
torch.nn.Module = type("Module", (), {"__init__": lambda self: None})

# Common nn modules
torch.nn.Linear = type(
    "Linear",
    (torch.nn.Module,),
    {
        "__init__": lambda self, in_features, out_features, bias=True: None,
        "forward": lambda self, x: x,
    },
)

torch.nn.ReLU = type(
    "ReLU",
    (torch.nn.Module,),
    {
        "__init__": lambda self, inplace=False: None,
        "forward": lambda self, x: x if hasattr(x, "__ge__") and x >= 0 else x,
    },
)

torch.nn.Embedding = type(
    "Embedding",
    (torch.nn.Module,),
    {
        "__init__": lambda self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
        dtype=None: None,
        "forward": lambda self, x: x,
    },
)

torch.nn.Dropout = type(
    "Dropout",
    (torch.nn.Module,),
    {"__init__": lambda self, p=0.5, inplace=False: None, "forward": lambda self, x: x},
)

torch.nn.LayerNorm = type(
    "LayerNorm",
    (torch.nn.Module,),
    {
        "__init__": lambda self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        device=None,
        dtype=None: None,
        "forward": lambda self, x: x,
    },
)

# Add functional module
torch.nn.functional = types.ModuleType("torch.nn.functional")
torch.nn.functional.relu = lambda x, inplace=False: x if hasattr(x, "__ge__") and x >= 0 else x
torch.nn.functional.dropout = lambda x, p=0.5, training=True, inplace=False: x
torch.nn.functional.softmax = lambda x, dim=None, _stacklevel=3, dtype=None: x
torch.nn.functional.gelu = lambda x: x

# Add autograd module
torch.autograd = types.ModuleType("torch.autograd")
torch.autograd.Variable = torch.Tensor  # Simple mock for Variable

torch.no_grad = lambda: contextlib.nullcontext()  # Simple mock for no_grad

# Add device class
torch.device = type(
    "device",
    (),
    {
        "__init__": lambda self, device, *args, **kwargs: setattr(self, "type", device),
        "__str__": lambda self: f"{self.type}",
        "__repr__": lambda self: f"device(type='{self.type}')",
    },
)

torch.cuda = create_module("torch.cuda")
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

torch.cuda.Stream = type(
    "Stream",
    (),
    {
        "__init__": lambda *args, **kwargs: None,
        "__enter__": lambda self: self,
        "__exit__": lambda *args: None,
        "synchronize": lambda: None,
    },
)
torch.cuda.current_stream = lambda device=None: torch.cuda.Stream()

torch.nn = create_module("torch.nn")
torch.nn.Module = type("Module", (), {"__init__": lambda *args, **kwargs: None})
torch.nn.Linear = type("Linear", (), {"__init__": lambda *args, **kwargs: None})

torch.nn.functional = create_module("torch.nn.functional")
torch.nn.functional.linear = lambda *args, **kwargs: torch.Tensor()
torch.nn.functional.relu = lambda *args, **kwargs: torch.Tensor()
torch.nn.functional.dropout = lambda *args, **kwargs: args[0] if args else torch.Tensor()
torch.nn.functional.layer_norm = lambda *args, **kwargs: args[0] if args else torch.Tensor()

torch.distributed = create_module("torch.distributed")
torch.distributed.get_rank = lambda: 0

# Create and register multiprocessing module
mp_module = create_module("torch.multiprocessing")
mp_module.Process = type(
    "Process",
    (),
    {
        "__init__": lambda self, *args, **kwargs: None,
        "start": lambda self: None,
        "join": lambda self, timeout=None: None,
        "is_alive": lambda self: False,
        "terminate": lambda self: None,
    },
)
mp_module.Queue = type(
    "Queue",
    (),
    {
        "__init__": lambda self, *args, **kwargs: None,
        "put": lambda self, item, *args, **kwargs: None,
        "get": lambda self, *args, **kwargs: None,
        "empty": lambda self: True,
    },
)

# Add torch.multiprocessing to sys.modules
torch.multiprocessing = mp_module
sys.modules["torch.multiprocessing"] = mp_module

# Now set up the rest of the mocks
import sys

transformers = sys.modules.get("transformers")
if transformers is None:
    import types as _types

    # Local placeholder to allow attribute assignments without touching sys.modules
    transformers = _types.SimpleNamespace()

transformers.__file__ = "/mock/transformers/__init__.py"
transformers.__version__ = "4.30.0"

# Import pydantic and its components
import pydantic
from pydantic import BaseModel, ValidationError, Field

# Add common classes and functions
transformers.PreTrainedModel = type(
    "PreTrainedModel",
    (object,),
    {
        "__init__": lambda self, *args, **kwargs: None,
        "config": {},
        "eval": lambda self: self,
        "train": lambda self, mode=True: self,
        "parameters": lambda self: [],
        "to": lambda self, *args, **kwargs: self,
        "save_pretrained": lambda self, *args, **kwargs: None,
    },
)

transformers.PretrainedConfig = type(
    "PretrainedConfig",
    (object,),
    {
        "__init__": lambda self, *args, **kwargs: None,
        "to_dict": lambda self: {},
        "from_pretrained": classmethod(lambda cls, *args, **kwargs: cls()),
    },
)

transformers.PreTrainedTokenizerBase = type(
    "PreTrainedTokenizerBase",
    (object,),
    {
        "__init__": lambda self, *args, **kwargs: None,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
        "unk_token_id": 3,
        "pad_token": "[PAD]",
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "name_or_path": "mock-model",
        "vocab_size": 30000,
        "model_max_length": 512,
        "is_fast": False,
        "padding_side": "right",
        "truncation_side": "right",
        "return_tensors": "pt",
        "pad": lambda self, *args, **kwargs: {"input_ids": [], "attention_mask": []},
        "encode_plus": lambda self, *args, **kwargs: {
            "input_ids": [],
            "attention_mask": [],
        },
        "decode": lambda self, *args, **kwargs: "decoded text",
        "batch_decode": lambda self, *args, **kwargs: ["decoded text"],
        "save_pretrained": lambda self, *args, **kwargs: None,
    },
)


# Add AutoModel classes
# Create a mock AutoModel class
class MockAutoModel:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return mock_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)


transformers.AutoModel = MockAutoModel

transformers.AutoModelForSequenceClassification = type(
    "AutoModelForSequenceClassification",
    (object,),
    {
        "from_pretrained": classmethod(lambda cls, *args, **kwargs: transformers.PreTrainedModel()),
    },
)


# Create a mock AutoModel class
class MockAutoModelForTokenClassification:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return mock_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)


transformers.AutoModelForTokenClassification = MockAutoModelForTokenClassification

transformers.AutoTokenizer = type(
    "AutoTokenizer",
    (object,),
    {
        "from_pretrained": classmethod(
            lambda cls, *args, **kwargs: transformers.PreTrainedTokenizerBase()
        ),
        "eos_token_id": 1,
        "bos_token_id": 2,
        "pad_token_id": 0,
        "unk_token_id": 3,
        "model_max_length": 512,
        "is_fast": False,
        "padding_side": "right",
        "truncation_side": "right",
        "__call__": lambda *args, **kwargs: {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        },
    },
)


# Create a mock config class
class MockConfig:
    def __init__(self, model_type="bert", **kwargs):
        self.model_type = model_type
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
        self.num_attention_heads = kwargs.get("num_attention_heads", 12)
        self.intermediate_size = kwargs.get("intermediate_size", 3072)
        self.hidden_act = kwargs.get("hidden_act", "gelu")
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.1)
        self.attention_probs_dropout_prob = kwargs.get("attention_probs_dropout_prob", 0.1)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 512)
        self.type_vocab_size = kwargs.get("type_vocab_size", 2)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-12)
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.position_embedding_type = kwargs.get("position_embedding_type", "absolute")
        self.use_cache = kwargs.get("use_cache", True)
        self.classifier_dropout = kwargs.get("classifier_dropout", None)

        # Tokenizer attributes
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.model_max_length = 512
        self.is_fast = False
        self.padding_side = "right"
        self.truncation_side = "right"

        # Add any additional attributes from kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# Create a mock AutoConfig class
class MockAutoConfig:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Check if we should return unused kwargs
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Create appropriate config based on model name
        model_name = str(pretrained_model_name_or_path).lower()

        # Filter out any kwargs that shouldn't be passed to the config
        config_kwargs = {
            k: v
            for k, v in kwargs.items()
            if not k.startswith("_") and k not in ["from_tf", "from_flax"]
        }

        # Create the appropriate config with the filtered kwargs
        if "gpt2" in model_name:
            config = MockGPT2Config(**config_kwargs)
        else:
            config = MockConfig(**config_kwargs)

        # If return_unused_kwargs is True, return a tuple of (config, unused_kwargs)
        if return_unused_kwargs:
            # Remove used kwargs from the original kwargs
            unused_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in config_kwargs or v != getattr(config, k, None)
            }
            return config, unused_kwargs

        # Otherwise, just return the config
        return config


transformers.AutoConfig = MockAutoConfig


# Create a mock model class
class MockPreTrainedModel:
    config_class = MockConfig

    def __init__(self, config=None, *args, **kwargs):
        self.config = config if config is not None else self.config_class()
        self.device = torch.device("cpu")
        self.training = False

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, *args, **kwargs):
        # Return a dummy output
        return {
            "logits": torch.randn(1, 10),
            "last_hidden_state": torch.randn(1, 10, self.config.hidden_size),
        }


# Update the transformers mocks
transformers.PreTrainedModel = MockPreTrainedModel


@pytest.fixture(autouse=True)
def _reset_model_registry_between_tests():
    """Ensure the global ModelRegistry is cleared between tests.

    This avoids singleton state leaking across tests which caused:
    - Duplicate registration ValueError mismatches
    - Missing metadata/ModelNotFoundError inconsistencies

    After clearing, we re-register default models to satisfy tests that
    assume defaults exist (e.g., generic BERT names).
    """
    try:
        # Import locally to respect patched modules set up in pytest_configure
        from medvllm.engine.model_runner.registry import get_registry

        reg = get_registry()
        # Clear all models and cache
        reg.clear()

        # Best-effort: restore defaults expected by some tests
        try:
            # Register standard defaults; method is resilient and logs on failure
            reg._register_default_models(force=False)  # type: ignore[attr-defined]
            # Also attempt to register medical defaults if available
            reg._register_medical_models()  # type: ignore[attr-defined]
        except Exception:
            # Don't fail the test setup if defaults cannot be registered in a mocked env
            pass
    except Exception:
        # If registry import fails for a subset of tests, just proceed
        pass
    yield


def mock_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    # For testing purposes, return a mock model with a config
    model_name = str(pretrained_model_name_or_path).lower()

    # Check if we should return unused kwargs (mimicking transformers' behavior)
    return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

    # Create appropriate config based on model name
    if "gpt2" in model_name:
        # Create a GPT2 config with provided kwargs
        config_kwargs = {
            k: v
            for k, v in kwargs.items()
            if not k.startswith("_") and k not in ["from_tf", "from_flax"]
        }
        config = MockGPT2Config(**config_kwargs)

        # Create appropriate model based on class name
        if hasattr(cls, "__name__") and "ForSequenceClassification" in cls.__name__:
            model = MockPreTrainedModel(config=config)
            model.__class__.__name__ = "GPT2ForSequenceClassification"
        elif hasattr(cls, "__name__") and "ForTokenClassification" in cls.__name__:
            model = MockPreTrainedModel(config=config)
            model.__class__.__name__ = "GPT2ForTokenClassification"
        else:
            model = MockPreTrainedModel(config=config)
            model.__class__.__name__ = "GPT2LMHeadModel"
    else:
        # Default to a basic model for other types
        config = MockConfig()
        model = MockPreTrainedModel(config=config)

    # Set device if specified
    device = kwargs.get("device", None)
    if device is not None:
        model.device = device

    # Add config to model
    model.config = config

    # Return either just the model or a tuple with unused kwargs
    if return_unused_kwargs:
        # Filter out used kwargs
        used_kwargs = set(config_kwargs.keys()) | {"device"}
        unused_kwargs = {k: v for k, v in kwargs.items() if k not in used_kwargs}
        return model, unused_kwargs

    return model, kwargs


transformers.PreTrainedModel.from_pretrained = classmethod(mock_from_pretrained)

# Add submodules
transformers.models = create_module("transformers.models")

# Create mock for transformers.models.gpt2
transformers.models.gpt2 = create_module("transformers.models.gpt2")

# Ensure the mock is in sys.modules
import sys

sys.modules["transformers.models.gpt2"] = transformers.models.gpt2
sys.modules["transformers.models.gpt2.configuration_gpt2"] = transformers.models.gpt2


# Mock GPT2Config
class MockGPT2Config(MockConfig):
    model_type = "gpt2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_embd = kwargs.get("n_embd", 768)
        self.n_layer = kwargs.get("n_layer", 12)
        self.n_head = kwargs.get("n_head", 12)
        self.n_positions = kwargs.get("n_positions", 1024)
        self.vocab_size = kwargs.get("vocab_size", 50257)
        self.bos_token_id = kwargs.get("bos_token_id", 50256)
        self.eos_token_id = kwargs.get("eos_token_id", 50256)
        self.n_ctx = kwargs.get("n_ctx", 1024)
        self.resid_pdrop = kwargs.get("resid_pdrop", 0.1)
        self.embd_pdrop = kwargs.get("embd_pdrop", 0.1)
        self.attn_pdrop = kwargs.get("attn_pdrop", 0.1)
        self.layer_norm_epsilon = kwargs.get("layer_norm_epsilon", 1e-5)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.summary_type = kwargs.get("summary_type", "cls_index")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Check if we should return unused kwargs
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Create a new instance of this config
        config = cls(**kwargs)

        # Return either just the config or a tuple with unused kwargs
        if return_unused_kwargs:
            return config, kwargs
        return config
        self.summary_use_proj = kwargs.get("summary_use_proj", True)
        self.summary_activation = kwargs.get("summary_activation", None)
        self.summary_proj_to_labels = kwargs.get("summary_proj_to_labels", True)
        self.summary_first_dropout = kwargs.get("summary_first_dropout", 0.1)

        # Update with any additional kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # Create a new instance with the config dictionary values
        return cls(**{**config_dict, **kwargs})

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Mock for from_pretrained that returns (config, kwargs) tuple."""
        config = cls()
        return config, kwargs

    def to_dict(self):
        # Convert the config to a dictionary
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __str__(self):
        return f"{self.__class__.__name__} {self.to_dict()}"


# Assign the mock config to the transformers module
transformers.models.gpt2.GPT2Config = MockGPT2Config

# Mock GPT2Model
transformers.models.gpt2.GPT2Model = type(
    "GPT2Model",
    (MockPreTrainedModel,),
    {
        "config_class": transformers.models.gpt2.GPT2Config,
        "base_model_prefix": "transformer",
        "main_input_name": "input_ids",
        "_no_split_modules": ["GPT2Block"],
        "_keys_to_ignore_on_load_missing": [r"attn.masked_bias", r"attn.bias"],
    },
)

# Mock GPT2LMHeadModel
transformers.models.gpt2.GPT2LMHeadModel = type(
    "GPT2LMHeadModel",
    (transformers.models.gpt2.GPT2Model,),
    {
        "tie_weights": lambda self: None,
        "get_output_embeddings": lambda self: None,
        "set_output_embeddings": lambda self, new_embeddings: None,
        "prepare_inputs_for_generation": lambda self, input_ids, past_key_values=None, **kwargs: {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        },
        "__module__": "transformers.models.gpt2.modeling_gpt2",
    },
)

# Mock GPT2ForSequenceClassification
transformers.models.gpt2.GPT2ForSequenceClassification = type(
    "GPT2ForSequenceClassification",
    (transformers.models.gpt2.GPT2Model,),
    {
        "num_labels": 2,
        "score": torch.nn.Linear(768, 2),
        "__module__": "transformers.models.gpt2.modeling_gpt2",
    },
)

# Mock GPT2ForTokenClassification
transformers.models.gpt2.GPT2ForTokenClassification = type(
    "GPT2ForTokenClassification",
    (transformers.models.gpt2.GPT2Model,),
    {
        "num_labels": 9,  # Typical number of NER labels
        "classifier": torch.nn.Linear(768, 9),
        "__module__": "transformers.models.gpt2.modeling_gpt2",
    },
)

# Mock the configuration module
transformers.models.gpt2.configuration_gpt2 = type(
    "GPT2ConfigModule",
    (),
    {
        "GPT2Config": transformers.models.gpt2.GPT2Config,
        "__all__": ["GPT2Config"],
        "__file__": "transformers/models/gpt2/configuration_gpt2.py",
    },
)

# Mock the modeling module
transformers.models.gpt2.modeling_gpt2 = type(
    "GPT2ModelingModule",
    (),
    {
        "GPT2Model": transformers.models.gpt2.GPT2Model,
        "GPT2LMHeadModel": transformers.models.gpt2.GPT2LMHeadModel,
        "GPT2ForSequenceClassification": transformers.models.gpt2.GPT2ForSequenceClassification,
        "GPT2ForTokenClassification": transformers.models.gpt2.GPT2ForTokenClassification,
        "__all__": [
            "GPT2Model",
            "GPT2LMHeadModel",
            "GPT2ForSequenceClassification",
            "GPT2ForTokenClassification",
        ],
        "__file__": "transformers/models/gpt2/modeling_gpt2.py",
    },
)

# Update sys.modules
sys.modules["transformers.models.gpt2.modeling_gpt2"] = transformers.models.gpt2.modeling_gpt2
sys.modules["transformers.models.gpt2.configuration_gpt2"] = (
    transformers.models.gpt2.configuration_gpt2
)


# Create a comprehensive mock for the transformers library
class MockTransformers:
    class AutoConfig:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            """Mock for AutoConfig.from_pretrained that returns (config, kwargs) tuple."""
            # Check if we should return unused kwargs
            return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

            # Create appropriate config based on model name
            model_name = str(pretrained_model_name_or_path).lower()

            # Filter out any kwargs that shouldn't be passed to the config
            config_kwargs = {
                k: v
                for k, v in kwargs.items()
                if not k.startswith("_") and k not in ["from_tf", "from_flax"]
            }

            if "gpt2" in model_name:
                config = MockGPT2Config(**config_kwargs)
            else:
                config = MockConfig(**config_kwargs)

            # Return either just the config or a tuple with unused kwargs
            if return_unused_kwargs:
                # Remove used kwargs from the original kwargs
                unused_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in config_kwargs or v != getattr(config, k, None)
                }
                return config, unused_kwargs

            return config

    class AutoModel:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            """Mock for AutoModel.from_pretrained that returns a mock model."""

            # Create a mock model
            class MockModel:
                def __init__(self, *args, **kwargs):
                    self.config = kwargs.get("config")
                    self.device = kwargs.get("device", "cpu")

                def to(self, device):
                    self.device = device
                    return self

            return MockModel()

    # Add configuration base class first
    class PretrainedConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            return cls(), kwargs

    # Store a reference to the config class
    _config_class = PretrainedConfig

    # Add model base class with config_class as a property
    class PreTrainedModel:
        @property
        @classmethod
        def config_class(cls):
            return _config_class

    class PreTrainedTokenizerBase:
        pass

    class PreTrainedTokenizer(PreTrainedTokenizerBase):
        pass

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            class MockTokenizer:
                def __call__(self, *args, **kwargs):
                    return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

                def encode(self, *args, **kwargs):
                    return [1, 2, 3]

                def decode(self, *args, **kwargs):
                    return "Mock decoded text"

            return MockTokenizer()

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return MockTransformers.AutoModel.from_pretrained(*args, **kwargs)

    class AutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return MockTransformers.AutoModel.from_pretrained(*args, **kwargs)

    class AutoModelForTokenClassification:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return MockTransformers.AutoModel.from_pretrained(*args, **kwargs)

    # Add models attribute with necessary submodules
    class Models:
        class gpt2:
            class GPT2Config:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    return MockGPT2Config(), kwargs

            class GPT2Model:
                pass

            class GPT2LMHeadModel:
                pass

            class GPT2ForSequenceClassification:
                pass

            class GPT2ForTokenClassification:
                pass

            # Add configuration and modeling modules
            configuration_gpt2 = type("configuration_gpt2", (), {"GPT2Config": GPT2Config})()
            modeling_gpt2 = type(
                "modeling_gpt2",
                (),
                {
                    "GPT2Model": GPT2Model,
                    "GPT2LMHeadModel": GPT2LMHeadModel,
                    "GPT2ForSequenceClassification": GPT2ForSequenceClassification,
                    "GPT2ForTokenClassification": GPT2ForTokenClassification,
                },
            )()

        # Add auto module
        class auto:
            class configuration_auto:
                @classmethod
                def AutoConfig(cls):
                    return MockTransformers.AutoConfig

            class modeling_auto:
                @classmethod
                def AutoModel(cls):
                    return MockTransformers.AutoModel

                @classmethod
                def AutoModelForCausalLM(cls):
                    return MockTransformers.AutoModelForCausalLM

                @classmethod
                def AutoModelForSequenceClassification(cls):
                    return MockTransformers.AutoModelForSequenceClassification

                @classmethod
                def AutoModelForTokenClassification(cls):
                    return MockTransformers.AutoModelForTokenClassification


# Create an instance of our mock transformers
mock_transformers = MockTransformers()

# Set up the models attribute and its submodules
mock_transformers.models = MockTransformers.Models()
mock_transformers.models.auto = MockTransformers.Models.auto()
mock_transformers.models.auto.configuration_auto = MockTransformers.Models.auto.configuration_auto()
mock_transformers.models.auto.modeling_auto = MockTransformers.Models.auto.modeling_auto()
mock_transformers.models.gpt2 = MockTransformers.Models.gpt2()

# Replace hard reassignment with safe augmentation only if a transformers module already exists.
# If it doesn't exist yet, defer to pytest_configure() which calls patch_transformers().
t = sys.modules.get("transformers")
if t is not None:
    # Ensure registry can detect test environment reliably
    try:
        setattr(t, "MockTransformers", True)
    except Exception:
        pass
    # Augment existing module with attributes from our mock without replacing identity
    for _attr in dir(mock_transformers):
        if _attr.startswith("__"):
            continue
        try:
            setattr(t, _attr, getattr(mock_transformers, _attr))
        except Exception:
            pass
    sys.modules["transformers"] = t
    transformers = t
else:
    # No transformers present yet; do not create or register one here.
    # pytest_configure will install the canonical MagicMock via patch_transformers().
    transformers = mock_transformers

# Re-export the mock classes for easier access
transformers.AutoConfig = MockTransformers.AutoConfig
transformers.AutoModel = MockTransformers.AutoModel
transformers.PreTrainedModel = MockTransformers.PreTrainedModel
transformers.AutoModelForCausalLM = MockTransformers.AutoModelForCausalLM
transformers.AutoModelForSequenceClassification = (
    MockTransformers.AutoModelForSequenceClassification
)
transformers.AutoModelForTokenClassification = MockTransformers.AutoModelForTokenClassification
transformers.models.auto = create_module("transformers.models.auto")
transformers.models.auto.configuration_auto = create_module(
    "transformers.models.auto.configuration_auto"
)

transformers.utils = create_module("transformers.utils")
transformers.utils.versions = type(
    "Versions", (), {"require_version": lambda *args, **kwargs: None}
)()
transformers.utils.logging = create_module("transformers.utils.logging")
transformers.utils.logging.get_logger = lambda name: MagicMock(
    debug=print, info=print, warning=print, error=print
)

transformers.tokenization_utils_base = create_module("transformers.tokenization_utils_base")
transformers.tokenization_utils_base.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase

transformers.configuration_utils = create_module("transformers.configuration_utils")
transformers.configuration_utils.PretrainedConfig = transformers.PreTrainedModel.config_class

transformers.modeling_utils = create_module("transformers.modeling_utils")
transformers.modeling_utils.PreTrainedModel = transformers.PreTrainedModel

# Add tree_map function
transformers.tree_map = lambda fn, *args, **kwargs: fn(*args) if args else None

# Add ModelOutput class
transformers.utils.ModelOutput = type(
    "ModelOutput",
    (dict,),
    {
        "__getattr__": lambda self, k: self[k],
        "__setattr__": lambda self, k, v: self.__setitem__(k, v),
    },
)

# pydantic is already imported at the top of the file

# Finalize and normalize a consistent torch stub to avoid conflicting mocks
import types as _types

# Build a clean torch module stub
_torch = _types.ModuleType("torch")
_torch.__version__ = "0.0.test"


# Dtypes as sentinel objects
class _DType:
    pass


_torch.dtype = _DType
_torch.float16 = _DType()
_torch.float32 = _DType()
_torch.float64 = _DType()
_torch.int64 = _DType()
_torch.qint8 = _DType()
_torch.quint8 = _DType()


class _Device:
    def __init__(self, t="cpu"):
        self.type = t

    def __str__(self):
        return self.type


class _Tensor:
    def __init__(self, shape=(1,), dtype=None, device=None):
        self._shape = tuple(shape)
        self.dtype = dtype if dtype is not None else _torch.float32
        self.device = device if device is not None else _Device("cpu")

    @property
    def shape(self):
        return self._shape

    def to(self, *args, **kwargs):
        # support device or dtype in .to()
        for a in args:
            if isinstance(a, str) and a in ("cpu", "cuda"):
                self.device = _Device(a)
        dtype = kwargs.get("dtype")
        if dtype is not None:
            self.dtype = dtype
        return self

    def cuda(self, *args, **kwargs):
        self.device = _Device("cuda")
        return self

    def cpu(self):
        self.device = _Device("cpu")
        return self

    # minimal in-place ops used by init code
    @property
    def data(self):
        return self

    def zero_(self):
        return self

    def __getitem__(self, idx):
        return self

    def __setitem__(self, key, value):
        return None

    def size(self, dim=None):
        if dim is None:
            return self._shape
        return self._shape[dim]

    def numel(self):
        # Product of shape dims; empty shape -> 1
        n = 1
        for d in self._shape:
            n *= max(1, int(d))
        return n


_torch.Tensor = _Tensor


def _infer_shape(data):
    # Infer shape from nested sequences like [[1,2,3]] -> (1,3)
    if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
        if len(data) == 0:
            return (0,)
        if hasattr(data[0], "__iter__"):
            return (len(data), len(data[0]))
        return (len(data),)
    return (1,)


def _tensor(data, *args, **kwargs):
    return _Tensor(
        shape=_infer_shape(data),
        dtype=kwargs.get("dtype", _torch.float32),
        device=_Device(kwargs.get("device", "cpu")),
    )


def _ones(*size, **kwargs):
    shape = size if size else (1,)
    return _Tensor(
        shape=shape,
        dtype=kwargs.get("dtype", _torch.float32),
        device=_Device(kwargs.get("device", "cpu")),
    )


def _zeros(*size, **kwargs):
    shape = size if size else (1,)
    return _Tensor(
        shape=shape,
        dtype=kwargs.get("dtype", _torch.float32),
        device=_Device(kwargs.get("device", "cpu")),
    )


_torch.tensor = _tensor
_torch.ones = _ones
_torch.zeros = _zeros
_torch.zeros_like = lambda t: _Tensor(shape=t.shape, dtype=t.dtype, device=t.device)


def _randn(*size, **kwargs):
    shape = size if size else (1,)
    return _Tensor(
        shape=shape,
        dtype=kwargs.get("dtype", _torch.float32),
        device=_Device(kwargs.get("device", "cpu")),
    )


_torch.randn = _randn


def _arange(start, end=None, step=1, **kwargs):
    if end is None:
        start, end = 0, start
    length = max(0, (end - start + (step - 1)) // step)
    return _Tensor(
        shape=(length,),
        dtype=kwargs.get("dtype", _torch.int64),
        device=_Device(kwargs.get("device", "cpu")),
    )


_torch.arange = _arange


def _noop_compile(fn=None, *args, **kwargs):
    if fn is None:

        def deco(f):
            return f

        return deco
    return fn


_torch.compile = _noop_compile


class _DeviceClass:
    def __init__(self, d):
        self.type = d

    def __str__(self):
        return self.type


_torch.device = _DeviceClass
_torch.no_grad = lambda: contextlib.nullcontext()

# cuda/distributed
_torch.cuda = _types.ModuleType("torch.cuda")
_torch.cuda.is_available = lambda: False
_torch.cuda.Stream = type("Stream", (), {})
_torch.cuda.set_per_process_memory_fraction = lambda fraction: None
_torch.cuda.memory = _types.ModuleType("torch.cuda.memory")
_torch.cuda.memory._set_allocator_settings = lambda settings: None
_torch.cuda.memory_allocated = lambda: 0
_torch.cuda.memory_reserved = lambda: 0
_torch.cuda.max_memory_allocated = lambda: 0
_torch.distributed = _types.ModuleType("torch.distributed")
_torch.distributed.is_available = lambda: False
_torch.distributed.is_initialized = lambda: False
_torch.distributed.init_process_group = lambda backend=None, rank=0, world_size=1: None
_torch.distributed.all_gather = lambda tensor_list, tensor: None

# torch.backends placeholders
_torch.backends = _types.ModuleType("torch.backends")
_torch.backends.cuda = _types.ModuleType("torch.backends.cuda")
_torch.backends.cuda.enable_flash_sdp = lambda flag=True: None
_torch.backends.cuda.matmul = _types.ModuleType("torch.backends.cuda.matmul")
_torch.backends.cuda.matmul.allow_tf32 = True
_torch.backends.cudnn = _types.ModuleType("torch.backends.cudnn")
_torch.backends.cudnn.allow_tf32 = True

# nn module with Embedding/Linear
_torch.nn = _types.ModuleType("torch.nn")


class _Module:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def parameters(self):
        return []

    def named_modules(self):
        return []


class _Embedding(_Module):
    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = _Tensor(
            shape=(num_embeddings, embedding_dim),
            dtype=_torch.float32,
            device=_Device("cpu"),
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x is a Tensor with shape (batch, seq)
        b = x.shape[0] if len(x.shape) > 0 else 1
        s = x.shape[1] if len(x.shape) > 1 else 1
        return _Tensor(
            shape=(b, s, self.embedding_dim),
            dtype=_torch.float32,
            device=_Device("cpu"),
        )


class _Linear(_Module):
    def __init__(self, in_features, out_features, bias=True, *args, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _Tensor(
            shape=(out_features, in_features),
            dtype=_torch.float32,
            device=_Device("cpu"),
        )
        self.bias = (
            _Tensor(shape=(out_features,), dtype=_torch.float32, device=_Device("cpu"))
            if bias
            else None
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # Replace last dim with out_features
        shp = tuple(x.shape)
        if len(shp) == 0:
            new_shape = (self.out_features,)
        else:
            new_shape = shp[:-1] + (self.out_features,)
        return _Tensor(
            shape=new_shape,
            dtype=_torch.float32,
            device=x.device if hasattr(x, "device") else _Device("cpu"),
        )


_torch.nn.Module = _Module
_torch.nn.Embedding = _Embedding
_torch.nn.Linear = _Linear


class _LayerNorm(_Module):
    def __init__(self, normalized_shape, eps=1e-5, *args, **kwargs):
        super().__init__()
        # Simplify: handle int shape
        dim = (
            normalized_shape
            if isinstance(normalized_shape, int)
            else (normalized_shape[-1] if hasattr(normalized_shape, "__iter__") else 1)
        )
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = _Tensor(shape=(dim,), dtype=_torch.float32, device=_Device("cpu"))
        self.bias = _Tensor(shape=(dim,), dtype=_torch.float32, device=_Device("cpu"))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return x


_torch.nn.LayerNorm = _LayerNorm


# optional conv placeholders for isinstance checks
class _ConvNd(_Module):
    pass


_torch.nn.Conv1d = _ConvNd
_torch.nn.Conv2d = _ConvNd
_torch.nn.Conv3d = _ConvNd


class _Parameter(_Tensor):
    def __init__(self, data=None, requires_grad=True):
        super().__init__(
            shape=getattr(data, "shape", (1,)),
            dtype=getattr(data, "dtype", _torch.float32),
            device=getattr(data, "device", _Device("cpu")),
        )
        self.requires_grad = requires_grad


_torch.nn.Parameter = _Parameter

# functional placeholder
_torch.nn.functional = _types.ModuleType("torch.nn.functional")
_torch.nn.functional.relu = lambda x, inplace=False: x
_torch.nn.functional.dropout = lambda x, p=0.5, training=True, inplace=False: x
_torch.nn.functional.softmax = lambda x, dim=None, _stacklevel=3, dtype=None: x
_torch.nn.functional.layer_norm = lambda x, normalized_shape=None, eps=1e-5: x

# basic math activations used by utils
_torch.tanh = lambda x: x
_torch.sigmoid = lambda x: x
_torch.pow = lambda x, y: x

# nn.init placeholder
_torch.nn.init = _types.ModuleType("torch.nn.init")
_torch.nn.init.normal_ = lambda tensor, mean=0.0, std=1.0: tensor
_torch.nn.init.constant_ = lambda tensor, val=0.0: tensor
_torch.nn.init.xavier_uniform_ = lambda tensor, gain=1.0: tensor

# tensor ops used by utils
_torch.narrow = lambda tensor, dim, start, length: tensor
_torch.cat = (
    lambda tensors, dim=0: tensors[0]
    if tensors
    else _Tensor(shape=(0,), dtype=_torch.float32, device=_Device("cpu"))
)

# Provide __spec__ so importlib.find_spec('torch') does not raise ValueError
import importlib.machinery as _machinery

try:
    _torch.__spec__ = _machinery.ModuleSpec("torch", loader=None)
except Exception:
    pass
for _name, _mod in [
    ("torch.nn", _torch.nn),
    ("torch.nn.functional", _torch.nn.functional),
    ("torch.cuda", _torch.cuda),
    ("torch.distributed", _torch.distributed),
    ("torch.backends", _torch.backends),
    ("torch.backends.cuda", _torch.backends.cuda),
    ("torch.backends.cudnn", _torch.backends.cudnn),
    ("torch.backends.cuda.matmul", _torch.backends.cuda.matmul),
]:
    try:
        _mod.__spec__ = _machinery.ModuleSpec(_name, loader=None)
    except Exception:
        pass

# Register the clean stub into sys.modules and export others
sys.modules.update(
    {
        "torch": _torch,
        "torch.nn": _torch.nn,
        "torch.nn.functional": _torch.nn.functional,
        "torch.cuda": _torch.cuda,
        "torch.distributed": _torch.distributed,
        "transformers": transformers,
        "transformers.models": transformers.models,
        "transformers.models.auto": transformers.models.auto,
        "transformers.models.auto.configuration_auto": transformers.models.auto.configuration_auto,
        "transformers.utils": transformers.utils,
        "transformers.utils.versions": transformers.utils.versions,
        "transformers.utils.logging": transformers.utils.logging,
        "transformers.tokenization_utils_base": transformers.tokenization_utils_base,
        "transformers.configuration_utils": transformers.configuration_utils,
        "transformers.modeling_utils": transformers.modeling_utils,
        "pydantic": pydantic,
        "pydantic.BaseModel": BaseModel,
        "pydantic.ValidationError": ValidationError,
        "pydantic.Field": Field,
    }
)


# Fixtures
@pytest.fixture
def mock_transformers():
    """Fixture that provides access to the mocked transformers module."""
    # Always return the live module from sys.modules to reflect any patches
    # applied in pytest_configure (e.g., patch_transformers()).
    return sys.modules.get("transformers", transformers)


@pytest.fixture(autouse=True)
def mock_torch(request):
    """Ensure the normalized torch stub is active for each test."""
    # Respect modules that install their own specialized torch mocks
    fspath = getattr(getattr(request, "node", None), "fspath", "")
    if fspath and "tests/unit/adapters/test_adapters_simple.py" in str(fspath):
        # Do not override; let the test's own mocks stand
        yield sys.modules.get("torch", _torch)
        return
    original = {
        "torch": sys.modules.get("torch"),
        "torch.nn": sys.modules.get("torch.nn"),
        "torch.nn.functional": sys.modules.get("torch.nn.functional"),
        "torch.cuda": sys.modules.get("torch.cuda"),
        "torch.distributed": sys.modules.get("torch.distributed"),
    }

    # Install the normalized stub
    sys.modules["torch"] = _torch
    sys.modules["torch.nn"] = _torch.nn
    sys.modules["torch.nn.functional"] = _torch.nn.functional
    sys.modules["torch.cuda"] = _torch.cuda
    sys.modules["torch.distributed"] = _torch.distributed

    yield _torch

    # Restore prior modules
    for k, v in original.items():
        if v is not None:
            sys.modules[k] = v
        else:
            sys.modules.pop(k, None)


@pytest.fixture(autouse=True)
def _ensure_transformers_meta():
    """Ensure mocked transformers has minimal module metadata.
    Some unit tests replace `transformers` with MagicMock lacking __file__.
    This fixture adds a synthetic __file__ so import tests don't fail.
    """
    m = sys.modules.get("transformers")
    if m is not None and not hasattr(m, "__file__"):
        try:
            m.__file__ = "/mock/transformers/__init__.py"
        except Exception:
            pass
    yield


@pytest.fixture
def mock_model_config():
    """Fixture that provides a mock model configuration."""
    return transformers.PreTrainedModel.config_class()


@pytest.fixture
def mock_tokenizer():
    """Fixture that provides a mock tokenizer."""
    return MagicMock()


@pytest.fixture
def temp_model_dir(tmp_path):
    """Fixture that provides a temporary directory for testing model configurations.

    This fixture creates a temporary directory with a basic model configuration file
    that can be used for testing model loading and configuration.
    """
    # Create a basic model configuration
    config = {
        "model_type": "medical_llm",
        "model_name_or_path": "test-model",
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "use_cache": True,
        "classifier_dropout": 0.1,
        "medical_specialties": ["cardiology", "radiology"],
        "anatomical_regions": ["head", "chest"],
        "max_sequence_length": 512,
    }

    # Create a config file in the temporary directory
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        import json

        json.dump(config, f)

    # Return the temporary directory path
    return tmp_path
