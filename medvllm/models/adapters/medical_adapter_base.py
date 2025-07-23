"""Base adapter class for medical models with optimized attention and layer implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

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
    get_activation_fn,
)
from medvllm.models.utils.attention_utils import (
    apply_attention,
)
from medvllm.models.utils.attention_utils import (
    get_attention_mask as get_attention_mask_from_utils,
)
from medvllm.models.utils.attention_utils import (
    get_extended_attention_mask as get_extended_attention_mask_from_utils,
)


class MedicalModelAdapterBase(nn.Module, ABC):
    """Base class for medical model adapters with KV cache optimizations.

    Features:
    - Efficient KV cache management for medical text
    - Support for distributed inference
    - Medical domain-specific optimizations
    - Integration with attention mechanisms
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        **kwargs
    ) -> None:
        """Initialize the medical model adapter with KV cache support.

        Args:
            model: The base model to adapt.
            config: Configuration dictionary for the adapter.
            **kwargs: Additional keyword arguments for medical-specific settings.
        """
        super().__init__()
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        # KV Cache Configuration
        self.kv_cache = None
        self.cache_enabled = config.get('use_kv_cache', True)
        self.cache_block_size = config.get('kv_cache_block_size', 256)
        self.max_cache_entries = config.get('max_kv_cache_entries', 1024)
        
        # Medical-specific settings
        self.medical_attention_window = config.get('medical_attention_window', 512)
        self.enable_medical_attention = config.get('enable_medical_attention', True)
        
        # Distributed training/inference settings
        self.tensor_parallel_size = config.get('tensor_parallel_size', 1)
        self.rank = config.get('rank', 0)
        self.world_size = config.get('world_size', 1)
        
        # Initialize KV cache if enabled
        if self.cache_enabled:
            self._initialize_kv_cache()

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

    def _initialize_kv_cache(self) -> None:
        """Initialize the KV cache with medical domain optimizations."""
        if not self.cache_enabled:
            return
            
        # Initialize KV cache with medical-specific parameters
        self.kv_cache = {
            'k_cache': None,
            'v_cache': None,
            'cache_size': 0,
            'max_size': self.max_cache_entries,
            'block_size': self.cache_block_size,
            'device': self.device,
            'dtype': self.dtype,
            'statistics': {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'total_used_blocks': 0,
                'total_blocks': 0
            }
        }
        
        # Allocate initial cache blocks
        self._allocate_cache_blocks()
    
    def _allocate_cache_blocks(self) -> None:
        """Allocate KV cache blocks based on configuration."""
        if not self.cache_enabled or self.kv_cache is None:
            return
            
        # Calculate number of blocks to allocate
        num_blocks = min(
            self.max_cache_entries,
            (self.config.max_sequence_length + self.cache_block_size - 1) // self.cache_block_size
        )
        
        if num_blocks <= 0:
            return
            
        # Allocate KV cache tensors
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        self.kv_cache['k_cache'] = torch.zeros(
            (num_blocks, num_heads, self.cache_block_size, head_dim),
            device=self.device,
            dtype=self.dtype
        )
        
        self.kv_cache['v_cache'] = torch.zeros_like(self.kv_cache['k_cache'])
        self.kv_cache['total_blocks'] = num_blocks
        self.kv_cache['free_blocks'] = list(range(num_blocks))
        self.kv_cache['allocated_blocks'] = {}
    
    def reset_cache(self) -> None:
        """Reset the KV cache while preserving the allocated memory."""
        if self.kv_cache is not None:
            # Reset cache state but keep the allocated memory
            self.kv_cache.update({
                'cache_size': 0,
                'free_blocks': list(range(self.kv_cache.get('total_blocks', 0))),
                'allocated_blocks': {},
                'statistics': {
                    'hits': 0,
                    'misses': 0,
                    'evictions': 0,
                    'total_used_blocks': 0,
                    'total_blocks': self.kv_cache.get('statistics', {}).get('total_blocks', 0)
                }
            })

    def to(self, device: Union[torch.device, str], *args, **kwargs) -> 'MedicalModelAdapterBase':
        """Move the adapter and KV cache to the specified device.

        Args:
            device: The target device.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The adapter instance.
        """
        self.device = torch.device(device)
        self.model = self.model.to(device, *args, **kwargs)
        
        # Move KV cache to device if it exists
        if self.kv_cache is not None:
            # Handle moving tensor components of the cache
            for key in ['k_cache', 'v_cache']:
                if key in self.kv_cache and self.kv_cache[key] is not None:
                    self.kv_cache[key] = self.kv_cache[key].to(device, *args, **kwargs)
            
            # Update device in cache metadata
            self.kv_cache['device'] = self.device
        
        return self
    
    def update_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the KV cache with new key and value states.
        
        Args:
            key_states: Key states to cache, shape [batch_size, num_heads, seq_len, head_dim]
            value_states: Value states to cache, shape [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Index of the transformer layer
            cache_position: Positions in the cache to update, if None appends to the end
            
        Returns:
            Tuple of (updated_key_states, updated_value_states)
        """
        if not self.cache_enabled or self.kv_cache is None:
            return key_states, value_states
            
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Initialize cache for this layer if needed
        if str(layer_idx) not in self.kv_cache['allocated_blocks']:
            self._allocate_layer_cache(layer_idx, batch_size, num_heads, head_dim)
            
        layer_cache = self.kv_cache['allocated_blocks'][str(layer_idx)]
        
        # Update cache with new key and value states
        if cache_position is None:
            # Append to the end of the cache
            new_cache_size = layer_cache['cache_size'] + seq_len
            
            # Resize cache if needed
            if new_cache_size > layer_cache['max_size']:
                self._resize_layer_cache(layer_idx, new_cache_size * 2)
                layer_cache = self.kv_cache['allocated_blocks'][str(layer_idx)]
                
            # Update cache position to the end
            cache_position = torch.arange(
                layer_cache['cache_size'],
                layer_cache['cache_size'] + seq_len,
                device=key_states.device
            )
            layer_cache['cache_size'] = new_cache_size
            
        # Update cache with new key and value states
        layer_cache['k_cache'][:, :, cache_position, :] = key_states.transpose(0, 1)
        layer_cache['v_cache'][:, :, cache_position, :] = value_states.transpose(0, 1)
        
        # Update statistics
        self.kv_cache['statistics']['total_used_blocks'] += seq_len
        
        return layer_cache['k_cache'], layer_cache['v_cache']
    
    def _allocate_layer_cache(
        self,
        layer_idx: int,
        batch_size: int,
        num_heads: int,
        head_dim: int
    ) -> None:
        """Allocate cache for a specific layer."""
        if self.kv_cache is None:
            return
            
        # Calculate required blocks
        blocks_needed = (self.max_cache_entries + self.cache_block_size - 1) // self.cache_block_size
        
        # Allocate blocks for this layer
        if len(self.kv_cache['free_blocks']) < blocks_needed:
            self._evict_blocks(blocks_needed - len(self.kv_cache['free_blocks']))
            
        # Get free blocks
        block_indices = self.kv_cache['free_blocks'][:blocks_needed]
        self.kv_cache['free_blocks'] = self.kv_cache['free_blocks'][blocks_needed:]
        
        # Initialize layer cache
        self.kv_cache['allocated_blocks'][str(layer_idx)] = {
            'k_cache': torch.zeros(
                (batch_size, num_heads, self.max_cache_entries, head_dim),
                device=self.device,
                dtype=self.dtype
            ),
            'v_cache': torch.zeros(
                (batch_size, num_heads, self.max_cache_entries, head_dim),
                device=self.device,
                dtype=self.dtype
            ),
            'block_indices': block_indices,
            'cache_size': 0,
            'max_size': self.max_cache_entries,
            'access_time': 0  # For LRU eviction
        }
    
    def _resize_layer_cache(self, layer_idx: int, new_size: int) -> None:
        """Resize the cache for a specific layer."""
        if self.kv_cache is None or str(layer_idx) not in self.kv_cache['allocated_blocks']:
            return
            
        layer_cache = self.kv_cache['allocated_blocks'][str(layer_idx)]
        old_size = layer_cache['max_size']
        
        if new_size <= old_size:
            return
            
        # Create new cache with increased size
        batch_size, num_heads, _, head_dim = layer_cache['k_cache'].shape
        
        new_k_cache = torch.zeros(
            (batch_size, num_heads, new_size, head_dim),
            device=self.device,
            dtype=self.dtype
        )
        new_v_cache = torch.zeros_like(new_k_cache)
        
        # Copy existing cache content
        new_k_cache[:, :, :old_size, :] = layer_cache['k_cache']
        new_v_cache[:, :, :old_size, :] = layer_cache['v_cache']
        
        # Update layer cache
        layer_cache['k_cache'] = new_k_cache
        layer_cache['v_cache'] = new_v_cache
        layer_cache['max_size'] = new_size
    
    def _evict_blocks(self, num_blocks: int) -> None:
        """Evict blocks from the cache using LRU policy."""
        if self.kv_cache is None or num_blocks <= 0:
            return
            
        # Sort layers by access time (oldest first)
        lru_layers = sorted(
            self.kv_cache['allocated_blocks'].items(),
            key=lambda x: x[1]['access_time']
        )
        
        evicted_blocks = 0
        
        # Evict blocks from oldest layers first
        for layer_id, layer_cache in lru_layers:
            if evicted_blocks >= num_blocks:
                break
                
            # Calculate blocks to evict from this layer
            blocks_to_evict = min(
                num_blocks - evicted_blocks,
                len(layer_cache['block_indices'])
            )
            
            if blocks_to_evict <= 0:
                continue
                
            # Add blocks to free list
            self.kv_cache['free_blocks'].extend(
                layer_cache['block_indices'][:blocks_to_evict]
            )
            
            # Update layer block indices
            layer_cache['block_indices'] = layer_cache['block_indices'][blocks_to_evict:]
            
            # Update statistics
            self.kv_cache['statistics']['evictions'] += blocks_to_evict
            evicted_blocks += blocks_to_evict

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
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        # Ensure model is on the correct device
        if self.model is None:
            raise ValueError("Model has not been initialized")

        self.model = self.model.to(self.device)

        # Default return_dict to True if not provided
        return_dict = return_dict if return_dict is not None else True

        # Initialize output containers with proper type hints
        all_hidden_states: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

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
        device = (
            input_ids.device
            if input_ids is not None
            else inputs_embeds.device if inputs_embeds is not None else self.device
        )

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

        # Initialize or update output containers
        if output_hidden_states:
            all_hidden_states = [embeddings]

        # Run through encoder layers
        hidden_states = embeddings
        for i, layer_module in enumerate(self.encoder):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # Update hidden states
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            # Update attentions
            if output_attentions and len(layer_outputs) > 1:
                all_attentions.append(layer_outputs[1])

        # Add last hidden state if not already added
        if (
            output_hidden_states
            and all_hidden_states
            and all_hidden_states[-1] is not hidden_states
        ):
            all_hidden_states.append(hidden_states)

        # Pool the output
        pooled_output = self.activation(self.pooler(hidden_states[:, 0]))

        # Return the output in the format expected by the parent class
        if not return_dict:
            output = [
                hidden_states,
                pooled_output,
                all_hidden_states if output_hidden_states else None,
                all_attentions if output_attentions else None,
            ]
            return tuple(v for v in output if v is not None)

        output_dict = {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled_output,
        }

        if output_hidden_states:
            output_dict["hidden_states"] = all_hidden_states
        if output_attentions:
            output_dict["attentions"] = all_attentions

        return output_dict

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
