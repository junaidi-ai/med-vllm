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
    
    # Add tensor function and basic tensor operations
    class MockTensor:
        def __init__(self, data):
            self.data = data
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Initialize shape based on the data
            self._shape = self._compute_shape(data)
            
        def _compute_shape(self, data):
            """Recursively compute the shape of the data."""
            # Handle scalar case
            if isinstance(data, (int, float, bool)):
                return (1,)
                
            # Handle empty list case
            if not isinstance(data, (list, tuple)):
                return (1,)  # Non-list data is treated as scalar
                
            if not data:
                return (0,)  # Truly empty list has shape (0,)
            
            # Handle nested lists
            shapes = []
            current = data
            while True:
                if not isinstance(current, (list, tuple)):
                    break
                shapes.append(len(current))
                if not current:  # Handle empty list in the hierarchy
                    break
                current = current[0]
            
            # Handle case where we have a list of non-sequences (1D tensor)
            if not shapes:
                return (1,)
                
            return tuple(shapes)
            
        def cpu(self):
            self.device = torch.device("cpu")
            return self
            
        def cuda(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return self
            
        def to(self, *args, **kwargs):
            # Handle device movement
            for arg in args:
                if isinstance(arg, (str, torch.device)):
                    self.device = torch.device(arg)
                    break
            # Handle device in kwargs
            if 'device' in kwargs:
                self.device = torch.device(kwargs['device'])
            return self
            
        def float(self):
            return self
            
        def long(self):
            return self
            
        def detach(self):
            return self
            
        def numpy(self):
            import numpy as np
            return np.array(self.data)
            
        def item(self):
            if isinstance(self.data, (list, tuple)) and len(self.data) == 1:
                return self.data[0] if not isinstance(self.data[0], (list, tuple)) else self.data[0][0]
            return self.data
            
        def unsqueeze(self, dim):
            # Create a new list with an extra dimension at the specified position
            if dim < 0:
                dim = len(self._shape) + dim + 1
            
            # Create new data with an extra dimension
            if dim == 0:
                new_data = [self.data]
                new_shape = (1,) + self._shape
            elif dim == len(self._shape):
                # Add new dimension at the end
                if len(self._shape) == 0:
                    new_data = [[x] for x in self.data] if isinstance(self.data, list) else [self.data]
                else:
                    new_data = [[x] for x in self.data] if self._shape[0] > 0 else []
                new_shape = self._shape + (1,)
            else:
                # For other dimensions, we need to recurse
                if not self._shape:
                    new_data = [self.data]
                    new_shape = (1,)
                else:
                    new_data = [MockTensor(x).unsqueeze(dim-1).data 
                              for x in self.data] if self._shape[0] > 0 else []
                    new_shape = self._shape[:dim] + (1,) + self._shape[dim:]
            
            result = MockTensor(new_data)
            result._shape = new_shape
            return result
                
        def squeeze(self, dim=None):
            if dim is None:
                # Remove all dimensions of size 1
                if not self._shape:
                    return self
                    
                # Find all dimensions that are not 1
                new_shape = [d for d in self._shape if d != 1]
                if not new_shape:
                    # If all dimensions were 1, return a scalar
                    return MockTensor(self.item())
                    
                # Create new data with squeezed dimensions
                new_data = self.data
                for d in reversed(range(len(self._shape))):
                    if self._shape[d] == 1:
                        if isinstance(new_data, list) and len(new_data) == 1:
                            new_data = new_data[0]
                
                result = MockTensor(new_data)
                result._shape = tuple(new_shape)
                return result
            else:
                # Remove specific dimension if its size is 1
                if dim < 0:
                    dim = len(self._shape) + dim
                if dim < 0 or dim >= len(self._shape):
                    raise IndexError(f"Dimension out of range (expected to be in range of [{-(len(self._shape))}, {len(self._shape)-1}], but got {dim})")
                
                if self._shape[dim] != 1:
                    return self
                    
                # Create new shape without the squeezed dimension
                new_shape = self._shape[:dim] + self._shape[dim+1:]
                
                # Create new data with the squeezed dimension
                def squeeze_dim(data, current_dim=0):
                    if current_dim == dim:
                        if isinstance(data, list) and len(data) == 1:
                            return data[0]
                        return data
                    if not isinstance(data, list):
                        return data
                    return [squeeze_dim(x, current_dim + 1) for x in data]
                
                new_data = squeeze_dim(self.data)
                result = MockTensor(new_data)
                result._shape = new_shape
                return result
                
        def dim(self):
            return len(self._shape)
            
        def size(self, dim=None):
            if dim is None:
                return self._shape
            if dim < 0:
                dim = len(self._shape) + dim
            if dim < 0 or dim >= len(self._shape):
                raise IndexError(f"Dimension out of range (expected to be in range of [{-(len(self._shape))}, {len(self._shape)-1}], but got {dim})")
            return self._shape[dim]
            
        @property
        def shape(self):
            return self._shape
            
        def __getitem__(self, idx):
            if not isinstance(idx, tuple):
                idx = (idx,)
            
            # Special case: CLS token selection with [:, 0]
            if (len(idx) == 2 and 
                isinstance(idx[0], slice) and 
                (idx[1] == 0 or idx[1] == slice(None, 1, None))):
                # For [:, 0] or [:, :1], return first token for each sequence in the batch
                batch_size = self._shape[0] if self._shape else 1
                hidden_size = self._shape[-1] if self._shape else 128
                new_shape = (batch_size, hidden_size)
                result_tensor = MockTensor([[0.0] * hidden_size for _ in range(batch_size)])
                result_tensor._shape = new_shape
                return result_tensor
            
            # Handle basic indexing for other cases
            try:
                result = self.data
                for i in idx:
                    if isinstance(i, slice):
                        result = result[i]
                    elif isinstance(i, int):
                        result = result[i] if i < len(result) else result[0]  # Handle out of bounds
                    else:
                        # Handle other index types (e.g., tensor)
                        result = result[i] if hasattr(result, '__len__') and i < len(result) else result[0]
            except (IndexError, TypeError, AttributeError):
                # If indexing fails, return a scalar with shape (1,)
                return MockTensor(0.0)
            
            # Create new shape based on the indexing
            if not hasattr(result, '__len__') or not result:
                # Scalar or empty result
                new_shape = (1,) if isinstance(result, (int, float)) else (0,)
            else:
                # Compute shape for non-empty sequences
                if isinstance(result[0], (int, float)):
                    new_shape = (len(result),)
                else:
                    # For nested sequences, compute shape recursively
                    new_shape = (len(result),) + self._compute_shape(result[0])
            
            # Create result tensor with correct shape
            result_tensor = MockTensor(result)
            result_tensor._shape = new_shape
            return result_tensor
            
        def __len__(self):
            return self._shape[0] if self._shape else 0
            
        def __iter__(self):
            # Return an iterator of MockTensors
            if not self._shape:
                return iter([])
            return (MockTensor(self.data[i]).squeeze(0) for i in range(self._shape[0]))
            
        def __contains__(self, item):
            return item in self.data
            
        def unsqueeze(self, dim):
            # Create a new list with an extra dimension at the specified position
            if dim < 0:
                dim = len(self._shape) + dim + 1
            
            # Create new data with an extra dimension
            if dim == 0:
                new_data = [self.data]
                new_shape = (1,) + self._shape
            elif dim == len(self._shape):
                # Add new dimension at the end
                if len(self._shape) == 0:
                    new_data = [[x] for x in self.data] if isinstance(self.data, list) else [self.data]
                else:
                    new_data = [[x] for x in self.data] if self._shape[0] > 0 else []
                new_shape = self._shape + (1,)
            else:
                # For other dimensions, we need to recurse
                if not self._shape:
                    new_data = [self.data]
                    new_shape = (1,)
                else:
                    new_data = [MockTensor(x).unsqueeze(dim-1).data 
                              for x in self.data] if self._shape[0] > 0 else []
                    new_shape = self._shape[:dim] + (1,) + self._shape[dim:]
            
            result = MockTensor(new_data)
            result._shape = new_shape
            return result
                
        def __add__(self, other):
            if isinstance(other, MockTensor):
                return MockTensor(self._process_nested(self.data, lambda x, y: x + y, other.data))
            return MockTensor(self._process_nested(self.data, lambda x, y: x + y, other))
            
        def __sub__(self, other):
            if isinstance(other, MockTensor):
                return MockTensor(self._process_nested(self.data, lambda x, y: x - y, other.data))
            return MockTensor(self._process_nested(self.data, lambda x, y: x - y, other))
            
        def _process_nested(self, data, op, other_data=None):
            if isinstance(data, list):
                if other_data is not None and isinstance(other_data, list):
                    return [self._process_nested(d, op, o) for d, o in zip(data, other_data)]
                return [self._process_nested(d, op, other_data) for d in data]
            elif other_data is not None:
                return op(data, other_data)
            return op(data)
            
            # Create a new list with an extra dimension at the specified position
            if dim < 0:
                dim = len(self.shape) + dim + 1
            
            if dim == 0:
                # Add new dimension at the beginning
                return MockTensor([self.data])
            elif dim == 1 and len(self.shape) == 1:
                # Add new dimension at position 1 for 1D tensors
                return MockTensor([[x] for x in self.data])
            else:
                # For other cases, just return a copy with the same data
                return MockTensor(self.data)
                
        def squeeze(self, dim=None):
            # Remove dimensions of size 1
            if dim is None:
                # Remove all dimensions of size 1
                if isinstance(self.data, list) and len(self.data) == 1:
                    if isinstance(self.data[0], list):
                        return MockTensor(self.data[0]).squeeze()
                    return MockTensor(self.data[0])
                return self
            else:
                # Remove specific dimension if its size is 1
                if dim < 0:
                    dim = len(self.shape) + dim
                if dim < 0 or dim >= len(self.shape):
                    raise IndexError(f"Dimension out of range (expected to be in range of [{-len(self.shape)}, {len(self.shape)-1}], but got {dim})")
                if self.shape[dim] == 1:
                    if len(self.shape) == 1:
                        return MockTensor(self.data[0] if isinstance(self.data, list) else self.data)
                    # For simplicity, just return a copy with the same data
                    return MockTensor(self.data)
                return self
                
        def to(self, *args, **kwargs):
            # For testing, just return self
            return self
            
        def _process_nested(self, data, func):
            """Helper method to process nested lists with a function."""
            if isinstance(data, list):
                return [self._process_nested(x, func) for x in data]
            return func(data)
            
        def __sub__(self, other):
            # Handle scalar subtraction (e.g., 1.0 - tensor)
            if isinstance(other, (int, float)):
                # Create a new MockTensor with the same shape but filled with the scalar value
                if isinstance(self.data, list):
                    # Handle nested lists of any depth
                    result = self._process_nested(self.data, lambda x: other - x)
                    return MockTensor(result)
                else:
                    # Scalar case
                    return MockTensor(other - self.data)
            # Handle tensor subtraction
            elif isinstance(other, MockTensor):
                # Simple element-wise subtraction for tensors of the same shape
                if isinstance(self.data, (list, int, float)) and isinstance(other.data, (list, int, float)):
                    # For testing, just return a new MockTensor with the same shape
                    if isinstance(self.data, list):
                        return MockTensor([[0.0] * len(row) if isinstance(row, list) else 0.0 for row in self.data])
                    return MockTensor(0.0)
            return self
            
        def __rsub__(self, other):
            # Handle right-side subtraction (e.g., 1.0 - tensor)
            # For right subtraction (other - self), we can reuse __sub__ with swapped operands
            if isinstance(other, (int, float)):
                if isinstance(self.data, list):
                    result = self._process_nested(self.data, lambda x: other - x)
                    return MockTensor(result)
                else:
                    return MockTensor(other - self.data)
            return self
            
        def __mul__(self, other):
            # Handle scalar multiplication
            if isinstance(other, (int, float)):
                if isinstance(self.data, list):
                    # Handle nested lists of any depth
                    result = self._process_nested(self.data, lambda x: x * other)
                    return MockTensor(result)
                else:
                    # Scalar case
                    return MockTensor(self.data * other)
            # Handle tensor multiplication
            elif isinstance(other, MockTensor):
                # For testing, just return a new MockTensor with the same shape
                if isinstance(self.data, list):
                    return MockTensor([[0.0] * len(row) if isinstance(row, list) else 0.0 for row in self.data])
                return MockTensor(0.0)
            return self
            
        def __rmul__(self, other):
            # Multiplication is commutative, so just call __mul__
            return self.__mul__(other)
    
    def tensor(data, *args, **kwargs):
        if isinstance(data, MockTensor):
            return data
        return MockTensor(data)

    torch.tensor = tensor
    torch.Tensor = type('Tensor', (object,), {
        'to': lambda self, *args, **kwargs: self,
        'cuda': lambda self, *args, **kwargs: self,
        'cpu': lambda self: self,
        'shape': property(lambda self: (1,)),
        'device': 'cpu',
        'dtype': torch.float32,
    })

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
