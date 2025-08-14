"""FlashAttention integration for medical models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


@dataclass
class FlashAttentionConfig:
    """Configuration for FlashAttention."""

    enable: bool = True
    causal: bool = True
    dropout: float = 0.0
    softmax_scale: Optional[float] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FlashAttentionConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enable": self.enable,
            "causal": self.causal,
            "dropout": self.dropout,
            "softmax_scale": self.softmax_scale,
        }


def _flash_attention_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[tuple] = None,
    output_attentions: bool = False,
    config: Optional[FlashAttentionConfig] = None,
) -> tuple:
    """Flash Attention forward pass."""
    if not FLASH_ATTN_AVAILABLE or not config or not config.enable:
        # Fall back to standard attention if FlashAttention is not available
        return self._original_forward(
            query,
            key,
            value,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

    # Handle different attention mask formats
    if attention_mask is not None:
        # Convert from {0, 1} to {True, False}
        attention_mask = attention_mask.squeeze(1).squeeze(
            1
        )  # [batch, 1, 1, seq_len] -> [batch, seq_len]
        attention_mask = attention_mask.bool()

        # Unpad input for better performance
        batch_size = query.shape[0]
        key_padding_mask = attention_mask

        # Reshape query, key, value
        query = query.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Unpad
        query_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query, key_padding_mask)
        key_unpad, _, cu_seqlens_k, max_seqlen_k = unpad_input(key, key_padding_mask)
        value_unpad, _, _, _ = unpad_input(value, key_padding_mask)

        # Reshape for FlashAttention
        query_unpad = query_unpad.unsqueeze(0)  # [1, num_tokens, num_heads, head_dim]
        key_unpad = key_unpad.unsqueeze(0)
        value_unpad = value_unpad.unsqueeze(0)

        # Apply FlashAttention
        dropout_p = config.dropout if self.training else 0.0
        attn_output = flash_attn_varlen_func(
            query_unpad,
            key_unpad,
            value_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=config.softmax_scale,
            causal=config.causal,
        )

        # Pad the output back to the original shape
        attn_output = pad_input(attn_output.squeeze(0), indices_q, batch_size, max_seqlen_q)
    else:
        # No padding, use standard FlashAttention
        query = query.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        dropout_p = config.dropout if self.training else 0.0
        attn_output = flash_attn_func(
            query,
            key,
            value,
            dropout_p=dropout_p,
            softmax_scale=config.softmax_scale,
            causal=config.causal,
        )

    # Reshape back to expected output shape
    attn_output = attn_output.transpose(
        1, 2
    )  # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]

    # Apply output projection
    attn_output = self.out_proj(attn_output)

    # Return output and attention weights (if needed)
    return (attn_output, None) if output_attentions else (attn_output,)


def enable_flash_attention(
    model: nn.Module, config: Optional[FlashAttentionConfig] = None, **kwargs
) -> nn.Module:
    """Enable FlashAttention for a transformer model.

    Args:
        model: The model to enable FlashAttention for
        config: FlashAttention configuration
        **kwargs: Override config values

    Returns:
        Model with FlashAttention enabled
    """
    if not FLASH_ATTN_AVAILABLE:
        print("Warning: FlashAttention not available. Using standard attention.")
        return model

    if config is None:
        config = FlashAttentionConfig(**kwargs)

    # Find and patch attention layers
    for name, module in model.named_modules():
        if hasattr(module, "_original_forward"):
            # Already patched
            continue

        # Patch different attention implementations
        if hasattr(module, "self"):
            # Standard attention (e.g., BERT, GPT)
            attention = module.self
            if hasattr(attention, "_original_forward"):
                continue

            # Save original forward
            attention._original_forward = attention.forward

            # Create a closure to maintain config
            def make_forward(attn, cfg):
                def forward(*args, **kwargs):
                    return _flash_attention_forward(attn, *args, config=cfg, **kwargs)

                return forward

            # Replace forward
            attention.forward = make_forward(attention, config)

        elif hasattr(module, "attn"):
            # GPT-2 style attention
            attention = module.attn
            if hasattr(attention, "_original_forward"):
                continue

            # Save original forward
            attention._original_forward = attention._attn

            # Create a closure to maintain config
            def make_attn(attn, cfg):
                def _attn(query, key, value, attention_mask=None, head_mask=None):
                    # Reshape for FlashAttention
                    batch_size = query.size(0)
                    q = query.view(batch_size, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
                    k = key.view(batch_size, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
                    v = value.view(batch_size, -1, attn.num_heads, attn.head_dim).transpose(1, 2)

                    # Apply FlashAttention
                    dropout_p = cfg.dropout if attn.training else 0.0
                    attn_output = flash_attn_func(
                        q,
                        k,
                        v,
                        dropout_p=dropout_p,
                        softmax_scale=cfg.softmax_scale,
                        causal=cfg.causal,
                    )

                    # Reshape back
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.view(batch_size, -1, attn.embed_dim)

                    # Apply output projection
                    attn_output = attn.out_proj(attn_output)
                    return attn_output, None

                return _attn

            # Replace attention function
            attention._attn = make_attn(attention, config)

    return model


def disable_flash_attention(model: nn.Module) -> nn.Module:
    """Disable FlashAttention and restore original attention.

    Args:
        model: Model with FlashAttention enabled

    Returns:
        Model with original attention
    """
    for module in model.modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            delattr(module, "_original_forward")

    return model
