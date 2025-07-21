"""Base adapter class for medical models with optimized attention and layer implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from medvllm.models.attention import MedicalMultiheadAttention
from medvllm.models.layers import (
    MedicalFeedForward,
    MedicalLayerNorm,
    MedicalTransformerEncoderLayer,
)
from medvllm.models.utils import (
    apply_attention,
    get_activation_fn,
    get_attention_mask,
)
from medvllm.models.utils import (
    get_extended_attention_mask as get_extended_attention_mask_from_utils,
)


class MedicalModelAdapterBase(nn.Module, ABC):
    """Base class for medical model adapters with optimized attention and layers.

    This class provides the foundation for medical model adapters with:
    - Optimized attention mechanisms
    - Specialized layer implementations
    - Memory-efficient architectures
    - Support for mixed precision training
    """

    def __init__(
        self, config: Dict, model: Optional[nn.Module] = None, **kwargs
    ) -> None:
        """Initialize the medical model adapter.

        Args:
            config: Configuration dictionary containing model hyperparameters
            model: Optional pre-initialized model to wrap
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model parameters from config
        self.hidden_size = config.get("hidden_size", 768)
        self.num_hidden_layers = config.get("num_hidden_layers", 12)
        self.num_attention_heads = config.get("num_attention_heads", 12)
        self.intermediate_size = config.get("intermediate_size", 3072)
        self.hidden_dropout_prob = config.get("hidden_dropout_prob", 0.1)
        self.attention_probs_dropout_prob = config.get(
            "attention_probs_dropout_prob", 0.1
        )
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-12)
        self.initializer_range = config.get("initializer_range", 0.02)
        self.vocab_size = config.get("vocab_size", 30522)
        self.type_vocab_size = config.get("type_vocab_size", 2)

        # Initialize components
        self._init_embeddings()
        self._init_encoder()
        self._init_pooler()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_embeddings(self) -> None:
        """Initialize the embedding layers."""
        self.word_embeddings = nn.Embedding(
            self.vocab_size, self.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            self.config.get("max_position_embeddings", 512), self.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            self.type_vocab_size, self.hidden_size
        )
        self.LayerNorm = MedicalLayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def _init_encoder(self) -> None:
        """Initialize the transformer encoder layers."""
        self.encoder = nn.ModuleList(
            [
                MedicalTransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=self.num_attention_heads,
                    dim_feedforward=self.intermediate_size,
                    dropout=self.hidden_dropout_prob,
                    activation=self.config.get("hidden_act", "gelu"),
                    layer_norm_eps=self.layer_norm_eps,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )

    def _init_pooler(self) -> None:
        """Initialize the pooler layer."""
        self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = get_activation_fn(self.config.get("hidden_act", "tanh"))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Ensure model is on the correct device
        if self.model is None:
            raise ValueError("Model has not been initialized")
            
        self.model = self.model.to(self.device)
        
        # Initialize output containers
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Default return_dict to True if not provided
        return_dict = return_dict if return_dict is not None else True
        
        """Forward pass for the medical model adapter.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            token_type_ids: Token type IDs of shape (batch_size, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len)
            head_mask: Mask for attention heads
            inputs_embeds: Precomputed embeddings of shape (batch_size, seq_len, hidden_size)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary output
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing the model outputs
        """
        # Handle input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Initialize position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # Initialize token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Add position and token type embeddings
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = get_extended_attention_mask_from_utils(
            attention_mask, (batch_size, seq_length), device
        )

        # Prepare head mask if needed
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config["num_hidden_layers"], -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer

        # Ensure all input tensors are on the correct device
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)
        if head_mask is not None:
            head_mask = head_mask.to(self.device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.device)

        # Initialize output containers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Run through encoder layers
        hidden_states = embeddings
        for i, layer_module in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Pool the output
        pooled_output = self.activation(self.pooler(hidden_states[:, 0]))

        # Return output
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    pooled_output,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )

        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled_output,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }

    @abstractmethod
    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings."""
        pass

    @abstractmethod
    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the input embeddings."""
        pass

    @abstractmethod
    def get_output_embeddings(self) -> nn.Module:
        """Get the output embeddings."""
        pass

    @abstractmethod
    def tie_weights(self) -> None:
        """Tie the weights between the input and output embeddings."""
        pass

    @abstractmethod
    def save_pretrained(self, save_directory: str) -> None:
        """Save the model to a directory."""
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs
    ) -> "MedicalModelAdapterBase":
        """Load a pretrained model."""
        pass
