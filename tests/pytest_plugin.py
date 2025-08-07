"""Pytest plugin to set up mocks before any test code is imported."""
import sys
import types
import contextlib

def pytest_configure():
    """Configure pytest and set up mocks before test collection."""
    # Create a mock torch module
    torch = types.ModuleType('torch')
    
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

    # Add dtypes
    torch.float32 = type('float32', (), {})
    torch.float16 = type('float16', (), {})
    torch.int64 = type('int64', (), {})
    
    # Add cuda module
    torch.cuda = types.ModuleType('torch.cuda')
    torch.cuda.is_available = lambda: False
    
    # Add device class
    class Device:
        def __init__(self, device_str):
            self.type = device_str.split(':')[0] if ':' in device_str else device_str
    
    torch.device = Device
    
    # Add Size class to support tensor.size()
    class Size(tuple):
        def __new__(cls, *args, **kwargs):
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                return super().__new__(cls, args[0])
            return super().__new__(cls, args)
            
        def __init__(self, *args, **kwargs):
            super().__init__()
            
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return Size(tuple(self)[idx])
            return super().__getitem__(idx)
            
        def __str__(self):
            return f'torch.Size({super().__str__()})'
            
        def __repr__(self):
            return self.__str__()
    
    torch.Size = Size
    
    # Add random number generation functions
    def randn(*size, **kwargs):
        # Return a MockTensor with the specified shape
        return MockTensor([0.0] * size[0] if len(size) == 1 else 
                         [[0.0] * size[1] for _ in range(size[0])] if len(size) == 2 else
                         [[[0.0] * size[2] for _ in range(size[1])] for _ in range(size[0])] if len(size) == 3 else
                         [0.0])
    
    torch.randn = randn
    
    # Add the tests directory to the Python path to ensure mock_models is importable
    import os
    import sys
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    
    # Import the unified MockTensor from mock_models
    # Import the mock models
    from .utils.mock_models import MockMedicalModel
    
    # Create a base tensor class that won't cause recursion
    class MockTensorBase:
        def __init__(self, *args, **kwargs):
            self.data = []
            self.shape = ()
            self.device = 'cpu'
            
        def to(self, *args, **kwargs):
            # Simple mock for the to() method
            if args and isinstance(args[0], (str, torch.device)):
                self.device = str(args[0])
            return self
            
        def __getitem__(self, idx):
            # Basic indexing support
            return self.data[idx] if hasattr(self.data, '__getitem__') else None
            
        def __setitem__(self, idx, value):
            # Basic item assignment support
            if hasattr(self.data, '__setitem__'):
                self.data[idx] = value
                
        def __len__(self):
            return len(self.data) if hasattr(self.data, '__len__') else 0
    
    # Create a factory function for our mock tensor
    def create_mock_tensor(*args, **kwargs):
        tensor = object.__new__(MockTensorBase)
        MockTensorBase.__init__(tensor, *args, **kwargs)
        return tensor
    
    # Set up mock implementations for torch functions
    class MockTorch:
        @staticmethod
        def tensor(data, *args, **kwargs):
            return create_mock_tensor(data, *args, **kwargs)
            
        @staticmethod
        def Tensor(*args, **kwargs):
            return create_mock_tensor(*args, **kwargs)
        
        @staticmethod
        def randn(*args, **kwargs):
            return create_mock_tensor(*args, **kwargs)
            
        @staticmethod
        def zeros(*args, **kwargs):
            return create_mock_tensor(*args, **kwargs)
            
        @staticmethod
        def ones(*args, **kwargs):
            return create_mock_tensor(*args, **kwargs)
            
        @staticmethod
        def FloatTensor(*args, **kwargs):
            return create_mock_tensor(*args, **kwargs)
            
        @staticmethod
        def LongTensor(*args, **kwargs):
            return create_mock_tensor(*args, **kwargs)
    
    # Apply the mocks
    torch.tensor = MockTorch.tensor
    torch.Tensor = MockTorch.Tensor
    torch.FloatTensor = MockTorch.FloatTensor
    torch.LongTensor = MockTorch.LongTensor
    
    # Add common tensor creation functions
    torch.randn = MockTorch.randn
    torch.zeros = MockTorch.zeros
    torch.ones = MockTorch.ones
    
    # Additional tensor operations that might be needed
    torch.empty = MockTorch.zeros  # Use zeros as a simple mock for empty
    torch.full = MockTorch.ones    # Use ones as a simple mock for full

    # Mock torch.nn.functional with all required functions
    def mock_relu(x):
        if isinstance(x, MockTensor):
            return x
        return x  # For real tensors, just pass through in mock
        
    torch.nn.functional = types.ModuleType('torch.nn.functional')
    torch.nn.functional.relu = mock_relu
    torch.nn.functional.dropout = lambda x, p=0.5, training=True, inplace=False: x
    torch.nn.functional.softmax = lambda x, dim=None: x
    torch.nn.functional.linear = lambda input, weight, bias=None: input
    torch.nn.functional.cross_entropy = lambda input, target, *args, **kwargs: tensor(0.0)
    
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
    
    # Add MultiheadAttention class
    class MultiheadAttention(torch.nn.Module):
        def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False, **kwargs):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.dropout = dropout
            self.batch_first = batch_first
            
        def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
            # For testing, just return a tuple with zeros for output and None for attention weights
            if isinstance(query, MockTensor):
                return (query, None)
            return (torch.zeros_like(query), None)
    
    # Add MultiheadAttention to torch.nn
    torch.nn.MultiheadAttention = MultiheadAttention
    
    # Add TransformerEncoderLayer class
    class TransformerEncoderLayer(torch.nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=False):
            super().__init__()
            self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
            self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
            self.dropout = torch.nn.Dropout(dropout)
            self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)
            self.dropout1 = torch.nn.Dropout(dropout)
            self.dropout2 = torch.nn.Dropout(dropout)
            self.batch_first = batch_first
            self.d_model = d_model
            
            if activation == "relu":
                self.activation = torch.nn.functional.relu
            else:
                self.activation = activation
                
        def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
            # For testing, return a tensor with the same shape as input
            if isinstance(src, MockTensor):
                # Get the input shape
                input_shape = src.shape
                
                # Ensure we have at least 3 dimensions (batch, seq, hidden)
                if len(input_shape) == 2:
                    # Add sequence dimension if missing
                    batch_size, hidden_size = input_shape
                    output_shape = (batch_size, 1, hidden_size)
                elif len(input_shape) == 3:
                    # Keep the same shape for 3D inputs
                    output_shape = input_shape
                else:
                    # For other cases, just return as is
                    return src
                
                # Create a new MockTensor with the correct shape
                # Fill with zeros for simplicity
                output_data = [[[0.0] * output_shape[-1] for _ in range(output_shape[1])] 
                             for _ in range(output_shape[0])]
                
                return MockTensor(output_data)
            
            # For non-MockTensor inputs (shouldn't happen in tests)
            return src
            
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
    
    torch.nn.TransformerEncoderLayer = TransformerEncoderLayer
    
    # Add Embedding class
    class Embedding(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, *args, **kwargs):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.weight = torch.randn(num_embeddings, embedding_dim)
            
        def forward(self, input):
            # For testing purposes, return a tensor with the expected shape
            if isinstance(input, MockTensor):
                # Get the input shape from MockTensor
                input_shape = input.shape
                if len(input_shape) == 1:
                    # Single sequence - add batch dimension
                    batch_size = 1
                    seq_length = input_shape[0]
                else:
                    # Already batched
                    batch_size = input_shape[0]
                    seq_length = input_shape[1] if len(input_shape) > 1 else 1
                
                # Create output with shape (batch_size, seq_length, embedding_dim)
                output_shape = (batch_size, seq_length, self.embedding_dim)
                output_data = [[[0.0] * self.embedding_dim for _ in range(seq_length)] 
                             for _ in range(batch_size)]
                return MockTensor(output_data)
            else:
                # For non-MockTensor inputs (shouldn't happen in tests)
                batch_size = input.size(0) if input.dim() > 1 else 1
                seq_length = input.size(-1) if input.dim() > 1 else input.size(0)
                return torch.randn(batch_size, seq_length, self.embedding_dim)
            
        def __call__(self, input):
            return self.forward(input)
    
    torch.nn.Embedding = Embedding
    
    # Add Linear class
    class Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = torch.randn(out_features, in_features)
            self.bias = torch.randn(out_features) if bias else None
            
        def forward(self, input):
            # For testing purposes, return a tensor with the expected shape
            if not isinstance(input, MockTensor):
                # For non-MockTensor inputs (shouldn't happen in tests)
                input_shape = input.shape if hasattr(input, 'shape') else (1,)
                
                # Special handling for pooler output
                if len(input_shape) == 2:  # [batch_size, hidden_size] - already pooled
                    output_shape = (input_shape[0], self.out_features)
                elif len(input_shape) == 3:  # [batch_size, seq_len, hidden_size]
                    if input_shape[1] == 1:  # This is likely the pooler input
                        output_shape = (input_shape[0], self.out_features)
                    else:
                        output_shape = (input_shape[0], input_shape[1], self.out_features)
                else:
                    output_shape = input_shape[:-1] + (self.out_features,)
                
                return torch.randn(*output_shape)
            
            # For MockTensor inputs
            input_shape = input.shape
            
            # Special handling for pooler output
            if len(input_shape) == 2:  # [batch_size, hidden_size] - already pooled
                output_shape = (input_shape[0], self.out_features)
            elif len(input_shape) == 3:  # [batch_size, seq_len, hidden_size]
                # Always use batch_size x out_features for pooler output
                # This ensures we get the right shape regardless of seq_len
                output_shape = (input_shape[0], self.out_features)
            else:
                # For other cases, just use the input dimensions and out_features
                output_shape = input_shape[:-1] + (self.out_features,)
            
            # Create output data with the correct shape
            if len(output_shape) == 2:
                # Pooled output: [batch_size, hidden_size]
                output_data = [[0.0] * output_shape[1] for _ in range(output_shape[0])]
            elif len(output_shape) == 3:
                # Sequence output: [batch_size, seq_len, hidden_size]
                output_data = [
                    [[0.0] * output_shape[2] for _ in range(output_shape[1])]
                    for _ in range(output_shape[0])
                ]
            else:
                # Fallback for other shapes
                output_data = [0.0] * output_shape[0] if output_shape else 0.0
            
            return MockTensor(output_data)
            
        def __call__(self, input):
            return self.forward(input)
    
    torch.nn.Linear = Linear
    
    # Add LayerNorm class
    torch.nn.LayerNorm = type('LayerNorm', (torch.nn.Module,), {
        '__init__': lambda self, normalized_shape, eps=1e-5, elementwise_affine=True: None,
        'forward': lambda self, x: x,
    })
    
    # First, properly implement the base Module class that ModuleList will inherit from
    class Module:
        def __init__(self):
            self._modules = {}
            self.training = True
        
        def add_module(self, name, module):
            self._modules[name] = module
            setattr(self, name, module)
        
        def __setattr__(self, name, value):
            if name != '_modules' and '_modules' in self.__dict__ and isinstance(value, Module):
                self._modules[name] = value
            object.__setattr__(self, name, value)
        
        def train(self, mode=True):
            self.training = mode
            for module in self._modules.values():
                module.train(mode)
            return self
        
        def eval(self):
            return self.train(False)
        
        def parameters(self, recurse=True):
            return []
        
        def to(self, *args, **kwargs):
            return self
        
        def state_dict(self, *args, **kwargs):
            return {}
        
        def load_state_dict(self, state_dict, strict=True):
            pass
            
        def apply(self, fn):
            # Apply fn to self
            fn(self)
            
            # Apply fn to all submodules
            for module in self._modules.values():
                if isinstance(module, Module):
                    module.apply(fn)
            return self
            
        def __call__(self, *args, **kwargs):
            # Default implementation that can be overridden by subclasses
            if hasattr(self, 'forward'):
                return self.forward(*args, **kwargs)
            raise NotImplementedError("Subclass must implement forward method")
    
    # Update the base Module class in torch.nn
    torch.nn.Module = Module
    
    # Now implement ModuleList with proper inheritance
    class ModuleList(Module):
        def __init__(self, modules=None):
            super().__init__()
            if modules is not None:
                for i, module in enumerate(modules):
                    self.add_module(str(i), module)
        
        def __iter__(self):
            return iter([getattr(self, str(i)) for i in range(len(self._modules))])
        
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return [getattr(self, str(i)) for i in range(len(self._modules))[idx]]
            # Handle negative indices
            if isinstance(idx, int) and idx < 0:
                idx = len(self._modules) + idx
                if idx < 0:
                    raise IndexError('list index out of range')
            return getattr(self, str(idx))
        
        def __len__(self):
            return len(self._modules)
        
        def append(self, module):
            self.add_module(str(len(self._modules)), module)
        
        def extend(self, modules):
            for module in modules:
                self.append(module)
    
    torch.nn.ModuleList = ModuleList
    
    # Add TransformerEncoder class
    class TransformerEncoder(torch.nn.Module):
        def __init__(self, encoder_layer, num_layers, norm=None):
            super().__init__()
            # Create a deep copy of the encoder layer for each layer
            self.layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(
                    d_model=encoder_layer.self_attn.embed_dim,
                    nhead=encoder_layer.self_attn.num_heads,
                    dim_feedforward=encoder_layer.linear1.out_features,
                    dropout=0.1,
                    activation='relu',
                    batch_first=encoder_layer.self_attn.batch_first
                ) for _ in range(num_layers)
            ])
            # Mark the last layer for special handling
            if len(self.layers) > 0:
                self.layers[-1]._is_last_layer = True
            self.norm = norm
            
        def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
            # Get input shape
            input_shape = src.shape if hasattr(src, 'shape') else (1, 5, 128)
            
            # For MockTensor inputs, ensure we have the expected shape
            if isinstance(src, MockTensor):
                if len(input_shape) == 2:
                    # Add sequence dimension if missing
                    batch_size, hidden_size = input_shape
                    output_shape = (batch_size, 1, hidden_size)
                elif len(input_shape) == 3:
                    # Keep the same shape for 3D inputs
                    output_shape = input_shape
                else:
                    # For other cases, just return as is
                    return src
                
                # Create output data with the correct shape
                output_data = [[[0.0] * output_shape[2] for _ in range(output_shape[1])] 
                             for _ in range(output_shape[0])]
                return MockTensor(output_data)
            
            # For non-MockTensor inputs (shouldn't happen in tests)
            return torch.randn(*input_shape)
    
    torch.nn.TransformerEncoder = TransformerEncoder

    # Functional module already defined above

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
