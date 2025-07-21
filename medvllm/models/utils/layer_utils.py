"""Utility functions for layer operations in medical models."""

import math
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

T = TypeVar("T")


def get_activation_fn(
    activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
) -> Callable[[Tensor], Tensor]:
    """Get the activation function by name.

    Args:
        activation: Name of the activation function (e.g., 'gelu', 'relu')

    Returns:
        The activation function
    """
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "silu" or activation == "swish":
        return F.silu
    elif activation == "gelu_new":
        return lambda x: F.gelu(x, approximate="tanh")
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    elif activation == "leaky_relu":
        return F.leaky_relu
    elif callable(activation):
        return activation
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


def create_initializer(initializer_range: float = 0.02) -> Callable[[nn.Module], None]:
    """Create a parameter initializer function.

    Args:
        initializer_range: Range for the uniform distribution

    Returns:
        A function that initializes parameters
    """

    def _initializer(module: nn.Module) -> None:
        """Initialize module parameters."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialize weights with a normal distribution
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    return _initializer


def get_parameter_dtype(module: nn.Module) -> torch.dtype:
    """Get the parameter dtype of a module.

    Args:
        module: The PyTorch module

    Returns:
        The dtype of the module's parameters
    """
    for param in module.parameters():
        return param.dtype
    return torch.float32  # Default if no parameters found


def create_sinusoidal_positional_embedding(
    num_positions: int,
    embedding_dim: int,
    padding_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Create sinusoidal positional embeddings.

    Args:
        num_positions: Maximum number of positions
        embedding_dim: Dimension of the embeddings
        padding_idx: If specified, pads the output with zeros at this index
        device: Device to create the embeddings on

    Returns:
        Positional embeddings of shape (num_positions, embedding_dim)
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)
    emb = torch.arange(num_positions, dtype=torch.float, device=device).unsqueeze(
        1
    ) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_positions, -1)

    if embedding_dim % 2 == 1:
        # Zero pad if embedding_dim is odd
        emb = torch.cat([emb, torch.zeros(num_positions, 1, device=device)], dim=1)

    if padding_idx is not None:
        emb[padding_idx] = 0.0

    return emb


def get_parameter_device(module: nn.Module) -> torch.device:
    """Get the device of a module's parameters.

    Args:
        module: The PyTorch module

    Returns:
        The device of the module's parameters
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        # For modules without parameters
        return torch.device("cpu")


def get_extended_attention_mask(
    attention_mask: Tensor,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
) -> Tensor:
    """Creates a causal attention mask for the transformer.

    Args:
        attention_mask: Attention mask of shape (batch_size, seq_len)
        input_shape: Shape of the input tensor (batch_size, seq_len, hidden_size)
        device: Device to create the mask on

    Returns:
        Extended attention mask of shape (batch_size, 1, 1, seq_len)
    """
    batch_size, seq_len = input_shape[0], input_shape[1]

    # Create a causal mask
    mask = torch.ones((batch_size, seq_len, seq_len), device=device, dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)

    # Expand the attention mask to account for the causal mask
    if attention_mask is not None:
        # Add a dimension for the heads
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=torch.float32)
        attention_mask = (1.0 - attention_mask) * -10000.0

        # Apply the causal mask
        mask = mask.unsqueeze(1)  # Add head dimension
        mask = mask.to(dtype=attention_mask.dtype, device=attention_mask.device)
        mask = (1.0 - mask) * -10000.0

        # Combine the masks
        mask = mask + attention_mask

    return mask


def create_position_ids_from_input_ids(
    input_ids: Tensor,
    padding_idx: int,
    past_key_values_length: int = 0,
) -> Tensor:
    """Create position ids from input ids.

    Args:
        input_ids: Input tensor of shape (batch_size, seq_len)
        padding_idx: Padding index
        past_key_values_length: Length of past key values (for generation)

    Returns:
        Position ids of shape (batch_size, seq_len)
    """
    mask = input_ids.ne(padding_idx).int()
    position_ids = torch.cumsum(mask, dim=1).long() - 1 + past_key_values_length
    position_ids.masked_fill_(mask == 0, padding_idx)
    return position_ids


def get_device_map(
    num_layers: int,
    num_devices: int,
    device_map: Optional[Dict[str, Union[int, str]]] = None,
) -> Dict[str, Union[int, str]]:
    """Get a device map for model parallelism.

    Args:
        num_layers: Number of layers in the model
        num_devices: Number of available devices
        device_map: Optional custom device map

    Returns:
        Device map dictionary
    """
    if device_map is not None:
        return device_map

    # Default to even distribution across devices
    layers_per_device = num_layers // num_devices
    device_map = {}

    for i in range(num_layers):
        device_id = i // layers_per_device
        if device_id >= num_devices:
            device_id = num_devices - 1
        device_map[f"layer_{i}"] = device_id

    return device_map


def get_parameter_count(module: nn.Module) -> int:
    """Get the total number of parameters in a module.

    Args:
        module: The PyTorch module

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
