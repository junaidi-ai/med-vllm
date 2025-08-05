"""
Attention utilities for medical language models.

This module provides various attention mechanisms and utilities used by the MedVLLM models.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[float] = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention as described in "Attention Is All You Need".
    
    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        value: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        dropout: Dropout probability
        
    Returns:
        Tuple of (output, attention_weights)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    
    if dropout is not None and dropout > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout)
    
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Split the last dimension into (num_heads, head_dim).
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, d_model)
        num_heads: Number of attention heads
        
    Returns:
        Tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size, seq_len, d_model = x.size()
    head_dim = d_model // num_heads
    
    # Reshape to (batch_size, seq_len, num_heads, head_dim)
    x = x.view(batch_size, seq_len, num_heads, head_dim)
    
    # Transpose to (batch_size, num_heads, seq_len, head_dim)
    return x.transpose(1, 2)


def combine_heads(x: torch.Tensor) -> torch.Tensor:
    """
    Combine the attention heads.
    
    Args:
        x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        
    Returns:
        Tensor of shape (batch_size, seq_len, d_model)
    """
    # Transpose to (batch_size, seq_len, num_heads, head_dim)
    x = x.transpose(1, 2)
    
    # Get dimensions
    batch_size, seq_len, num_heads, head_dim = x.size()
    d_model = num_heads * head_dim
    
    # Reshape to (batch_size, seq_len, d_model)
    return x.contiguous().view(batch_size, seq_len, d_model)


def create_attention_mask(
    input_ids: torch.Tensor, 
    padding_idx: int = 0
) -> torch.Tensor:
    """
    Create attention mask for padded tokens.
    
    Args:
        input_ids: Input token IDs of shape (batch_size, seq_len)
        padding_idx: Padding token index
        
    Returns:
        Attention mask of shape (batch_size, 1, 1, seq_len)
    """
    # Create mask for padding tokens (1 for real tokens, 0 for padding)
    mask = (input_ids != padding_idx).unsqueeze(1).unsqueeze(2)
    return mask.to(dtype=torch.float32)


def get_extended_attention_mask(
    attention_mask: torch.Tensor,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    
    Args:
        attention_mask: Tensor with mask values of shape (batch_size, seq_len)
        input_shape: Tuple of (batch_size, seq_len)
        dtype: Data type for the output tensor
        
    Returns:
        Extended attention mask of shape (batch_size, 1, 1, seq_len)
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    return extended_attention_mask
