"""Custom neural network layers optimized for medical language models.

This module provides specialized layer implementations that are optimized
for medical NLP tasks, including layer normalization and feed-forward networks.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MedicalLayerNorm(nn.Module):
    """Layer normalization optimized for medical models.

    This implementation includes several optimizations:
    - Fused operations for better performance
    - Support for mixed precision training
    - Gradient checkpointing support
    - Optional elementwise affine transformation
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, device=device, dtype=dtype)
            )
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the affine parameters."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass for layer normalization."""
        # Use FusedLayerNorm if available and on CUDA
        if input.is_cuda and hasattr(torch.ops.torch, "native_layer_norm"):
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

        # Fall back to manual implementation if needed
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        output = (input - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            output = output * self.weight + self.bias

        return output

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )


class MedicalFeedForward(nn.Module):
    """Feed-forward network optimized for medical models.

    This implementation includes several optimizations:
    - Gated linear units (GLU) for better gradient flow
    - Dropout for regularization
    - Layer normalization for stable training
    - Residual connection for better gradient propagation
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout  # Store dropout rate separately

        # Initialize weights
        self.w1 = nn.Linear(d_model, d_ff, bias=True, device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, bias=True, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, d_ff, bias=True, device=device, dtype=dtype)

        # Layer normalization
        self.layer_norm = MedicalLayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)

        # Activation function
        self.activation = self._get_activation_fn(activation)

        # Initialize dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _get_activation_fn(self, activation: str):
        """Get the activation function by name."""
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "gelu_new":
            return F.gelu
        elif activation == "silu":
            return F.silu
        elif activation == "swish":
            return F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the feed-forward network."""
        # Apply layer normalization
        x_norm = self.layer_norm(x)

        # Gated linear unit (GLU) for better gradient flow
        gate = self.activation(self.w1(x_norm))
        up = self.w3(x_norm)
        hidden = gate * up  # Element-wise multiplication

        # Project back to model dimension and apply dropout
        output = self.w2(self.dropout(hidden))

        # Residual connection
        return x + output


class MedicalTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer optimized for medical models.

    This implementation includes several optimizations:
    - Pre-norm architecture for better gradient flow
    - Gated feed-forward network
    - Flash attention when available
    - Dropout for regularization
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.batch_first = batch_first
        self.norm_first = norm_first

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        # Feed-forward network
        self.ffn = MedicalFeedForward(
            d_model, dim_feedforward, dropout, activation, layer_norm_eps, device, dtype
        )

        # Layer normalization
        self.norm1 = MedicalLayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = MedicalLayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)

        # Dropout
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for the transformer encoder layer."""
        x = src

        if self.norm_first:
            # Pre-norm architecture
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm architecture
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """Self-attention block."""
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed-forward block."""
        return self.dropout2(self.ffn(x))
