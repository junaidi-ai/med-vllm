"""
Utility functions for neural network layers.

This module provides various utility functions for working with neural network layers,
including weight initialization, layer normalization, and other common operations.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """Initialize the weights of a module.

    Args:
        module: The module to initialize.
        std: Standard deviation for normal initialization.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit (GELU) activation function.

    For information: OpenAI's GPT uses GELU with an approximation.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        x: Input tensor.

    Returns:
        Output tensor with GELU activation applied.
    """
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GELU implementation that matches the TensorFlow version.

    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415

    Args:
        x: Input tensor.

    Returns:
        Output tensor with GELU activation applied.
    """
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function.

    Reference: https://arxiv.org/abs/1710.05941

    Args:
        x: Input tensor.

    Returns:
        Output tensor with Swish activation applied.
    """
    return x * torch.sigmoid(x)


ACT2FN = {
    "relu": F.relu,
    "gelu": gelu,
    "gelu_new": gelu_new,
    "swish": swish,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


def get_activation_fn(activation_string: str) -> callable:
    """Get activation function by name.

    Args:
        activation_string: Name of the activation function.

    Returns:
        The activation function.

    Raises:
        KeyError: If the activation function is not found.
    """
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            f"Function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}"
        )


def create_linear_layer(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    """Create a linear layer with appropriate initialization.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to False, the layer will not learn an additive bias.

    Returns:
        Initialized linear layer.
    """
    layer = nn.Linear(in_features, out_features, bias=bias)
    # Apply custom initialization
    nn.init.xavier_uniform_(layer.weight)
    if bias:
        nn.init.constant_(layer.bias, 0.0)
    return layer


def create_embedding_layer(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: Optional[int] = None,
    std: float = 0.02,
) -> nn.Embedding:
    """Create an embedding layer with appropriate initialization.

    Args:
        num_embeddings: Size of the dictionary of embeddings.
        embedding_dim: The size of each embedding vector.
        padding_idx: If specified, the entries at padding_idx do not contribute to the gradient.
        std: Standard deviation for normal initialization.

    Returns:
        Initialized embedding layer.
    """
    embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(embedding.weight, mean=0.0, std=std)
    if padding_idx is not None:
        embedding.weight.data[padding_idx].zero_()
    return embedding


def create_layer_norm(
    normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5
) -> nn.LayerNorm:
    """Create a LayerNorm module with appropriate initialization.

    Args:
        normalized_shape: Input shape from an expected input of size
            `[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`.
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: A value added to the denominator for numerical stability.

    Returns:
        Initialized LayerNorm module.
    """
    return nn.LayerNorm(normalized_shape, eps=eps)


def get_parameter_dtype(module: nn.Module) -> torch.dtype:
    """Get the parameter dtype of a module.

    Args:
        module: The module to get the parameter dtype from.

    Returns:
        The dtype of the first parameter in the module, or torch.float32 if no parameters exist.
    """
    for param in module.parameters():
        return param.dtype
    return torch.float32
