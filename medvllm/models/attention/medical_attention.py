"""Custom attention mechanisms for medical language models.

This module provides optimized attention implementations specifically designed
for medical NLP tasks, with support for various attention patterns and optimizations.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MedicalMultiheadAttention(nn.Module):
    """Multi-head attention implementation optimized for medical models.

    This implementation includes optimizations for medical NLP tasks and
    supports both regular and flash attention when available.
    """

    bias_k: Optional[torch.Tensor]  # Type annotation for mypy
    bias_v: Optional[torch.Tensor]  # Type annotation for mypy

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} "
                f"and `num_heads`: {num_heads})."
            )

        # Initialize projection layers
        if self._qkv_same_embed_dim:
            # Self-attention case
            self.in_proj_weight = nn.Parameter(
                torch.empty((3 * embed_dim, embed_dim), device=device, dtype=dtype)
            )
            self.register_parameter("in_proj_bias", None)
        else:
            # Cross-attention case
            self.q_proj_weight = nn.Parameter(
                torch.empty((embed_dim, embed_dim), device=device, dtype=dtype)
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.kdim), device=device, dtype=dtype)
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.vdim), device=device, dtype=dtype)
            )
            self.register_parameter("in_proj_weight", None)

        # Output projection
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        # Optional bias for key and value
        if add_bias_kv:
            self.bias_k = nn.Parameter(
                torch.empty((1, 1, embed_dim), device=device, dtype=dtype)
            )
            self.bias_v = nn.Parameter(
                torch.empty((1, 1, embed_dim), device=device, dtype=dtype)
            )
        else:
            # Initialize as None and let mypy know these will be Optional[Parameter]
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters following Xavier/Glorot initialization."""
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass for multi-head attention.

        Args:
            query: Query tensor of shape (L, N, E) or (N, L, E) if batch_first=True
            key: Key tensor of shape (S, N, E_k) or (N, S, E_k)
            value: Value tensor of shape (S, N, E_v) or (N, S, E_v)
            key_padding_mask: Mask for padding tokens of shape (N, S)
            need_weights: Whether to return attention weights
            attn_mask: 2D or 3D mask for attention
            average_attn_weights: Whether to average attention weights across heads

        Returns:
            Tuple of (output, attention_weights)
        """
        # Handle batch dimension
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Project query, key, value
        if self._qkv_same_embed_dim:
            # Self-attention case
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(
                3, dim=-1
            )
        else:
            # Cross-attention case
            q = F.linear(query, self.q_proj_weight, None)
            k = F.linear(key, self.k_proj_weight, None)
            v = F.linear(value, self.v_proj_weight, None)

        # Add bias if needed - use type: ignore for mypy since we've already checked for None
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(1, k.size(1), 1)])  # type: ignore[union-attr]
            v = torch.cat([v, self.bias_v.repeat(1, v.size(1), 1)])  # type: ignore[union-attr]
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # Reshape for multi-head attention
        q = (
            q.contiguous()
            .view(q.size(0), -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            k.contiguous()
            .view(k.size(0), -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            v.contiguous()
            .view(v.size(0), -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scale the dot product
        scaling = float(self.head_dim) ** -0.5
        q = q * scaling

        # Compute attention scores
        attn_output_weights = torch.matmul(q, k.transpose(-2, -1))

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                -1, self.num_heads, q.size(2), k.size(2)
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(-1, q.size(2), k.size(2))

        # Apply softmax to get attention weights
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        # Apply attention weights to values
        attn_output = torch.bmm(attn_output_weights, v)

        # Reshape and project back to embedding dimension
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(-1, self.num_heads * self.head_dim)
        )
        attn_output = self.out_proj(attn_output)

        # Reshape back to (N, L, E) if batch_first
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        # Average attention weights if requested
        if need_weights:
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            return attn_output, attn_output_weights

        return attn_output, None


def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    return_attn_probs: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention forward pass if available, otherwise fall back to PyTorch's implementation.

    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        value: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        dropout_p: Dropout probability
        scale: Scale factor for the dot product
        softmax_scale: Scale factor for softmax
        attn_mask: Optional attention mask
        is_causal: Whether to use causal attention
        return_attn_probs: Whether to return attention probabilities

    Returns:
        Tuple of (output, attention_probs)
    """
    try:
        from flash_attn import flash_attn_func

        # Use Flash Attention if available
        output = flash_attn_func(
            query,
            key,
            value,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=is_causal,
            return_attn_probs=return_attn_probs,
        )

        if return_attn_probs:
            output, attn_probs = output
            return output, attn_probs
        return output, None

    except ImportError:
        # Fall back to PyTorch's implementation
        if scale is None:
            scale = 1.0 / (query.size(-1) ** 0.5)

        # Compute attention scores
        attn_scores = torch.matmul(query * scale, key.transpose(-2, -1))

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # Apply attention weights to values
        output = torch.matmul(attn_weights, value)

        if return_attn_probs:
            return output, attn_weights
        return output, None
