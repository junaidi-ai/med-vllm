"""Utility functions for attention mechanisms in medical models."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_attention_mask(
    attention_mask: Optional[Tensor], input_shape: Tuple[int, ...]
) -> Optional[Tensor]:
    """Create attention mask from input mask.

    Args:
        attention_mask: Attention mask of shape (batch_size, seq_len) or (batch_size, seq_len, seq_len)
        input_shape: Shape of the input tensor (batch_size, seq_len, hidden_size)

    Returns:
        Attention mask of shape (batch_size, 1, seq_len, seq_len) or None if no mask is provided
    """
    if attention_mask is None:
        return None

    # Handle 2D attention mask (batch_size, seq_len)
    if attention_mask.dim() == 2:
        batch_size, seq_len = attention_mask.shape
        # Expand to (batch_size, 1, 1, seq_len)
        attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
        # Create causal mask if needed
        attention_mask = attention_mask.to(dtype=torch.float32)
        attention_mask = (1.0 - attention_mask) * -10000.0
    # Handle 3D attention mask (batch_size, seq_len, seq_len)
    elif attention_mask.dim() == 3:
        batch_size, seq_len, _ = attention_mask.shape
        # Expand to (batch_size, 1, seq_len, seq_len)
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.to(dtype=torch.float32)

    return attention_mask


def apply_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Apply attention mechanism with optional masking and dropout.

    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        value: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        attention_mask: Optional attention mask of shape (batch_size, seq_len) or (batch_size, seq_len, seq_len)
        dropout_p: Dropout probability
        is_causal: Whether to use causal attention
        scale: Scaling factor for the dot product

    Returns:
        Tuple of (output, attention_weights)
    """
    # Scale the dot product
    if scale is None:
        scale = 1.0 / (query.size(-1) ** 0.5)

    # Compute attention scores
    attn_scores = torch.matmul(query * scale, key.transpose(-2, -1))

    # Apply attention mask if provided
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(~attention_mask, float("-inf"))
        else:
            attn_scores = attn_scores + attention_mask

    # Apply causal mask if needed
    if is_causal:
        seq_len = query.size(-2)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    # Apply softmax to get attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0 and attn_weights is not None:
        attn_weights = F.dropout(
            attn_weights, p=dropout_p, training=attn_weights.requires_grad
        )

    # Apply attention weights to values
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


def split_heads(tensor: Tensor, num_heads: int) -> Tensor:
    """Split the last dimension into (num_heads, head_dim)."""
    batch_size, seq_len, hidden_size = tensor.size()
    head_dim = hidden_size // num_heads
    return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def combine_heads(tensor: Tensor) -> Tensor:
    """Combine attention head outputs."""
    batch_size, num_heads, seq_len, head_dim = tensor.size()
    hidden_size = num_heads * head_dim
    return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)


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
