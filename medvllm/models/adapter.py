"""Adapter interface for medical language models.

This module provides a flexible adapter interface to integrate various medical language models
with the Nano vLLM architecture, ensuring consistent behavior and optimization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class MedicalModelAdapter(ABC, nn.Module):
    """Abstract base class for medical model adapters.

    This class defines the interface that all medical model adapters must implement
    to be compatible with the Nano vLLM architecture.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize the adapter with a model and configuration.

        Args:
            model: The underlying model to adapt
            config: Configuration dictionary for the adapter
        """
        super().__init__()
        self.model = model
        self.config = config
        self.kv_cache: Optional[Dict[str, torch.Tensor]] = None
        self.cuda_graphs: Optional[Any] = None

    @abstractmethod
    def setup_for_inference(self, **kwargs) -> None:
        """Prepare the model for inference with optimizations.

        This should be called before any inference to set up CUDA graphs,
        KV cache, and other optimizations.
        """
        pass

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments for the forward pass

        Returns:
            Model outputs
        """
        pass

    def reset_cache(self) -> None:
        """Reset the KV cache if it exists."""
        if self.kv_cache is not None:
            self.kv_cache = None

    def to(self, device: torch.device) -> "MedicalModelAdapter":
        """Move the model to the specified device."""
        self.model = self.model.to(device)
        return self


class BioBERTAdapter(MedicalModelAdapter):
    """Adapter for BioBERT models optimized for medical NLP tasks.

    This adapter handles:
    - KV caching for efficient inference
    - Weight conversion from Hugging Face format
    - Special handling of biomedical tokens and embeddings
    - CUDA graph optimization
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize the BioBERT adapter.

        Args:
            model: The underlying BioBERT model
            config: Configuration dictionary with model parameters
        """
        super().__init__(model, config)
        self.model_type = "biobert"
        self.num_hidden_layers = getattr(model.config, "num_hidden_layers", 12)
        self.num_attention_heads = getattr(model.config, "num_attention_heads", 12)
        self.hidden_size = getattr(model.config, "hidden_size", 768)
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Initialize tokenizer for BioBERT-specific handling
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._setup_biobert_tokenizer()
        
        # Initialize weights and handle biomedical embeddings
        self._initialize_weights()
        self._setup_biomedical_embeddings()

    def _setup_biobert_tokenizer(self) -> None:
        """Set up BioBERT-specific tokenizer with biomedical vocabulary."""
        try:
            # Try to get tokenizer from model config or create default
            model_name = getattr(self.model.config, '_name_or_path', 'dmis-lab/biobert-v1.1')
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                do_lower_case=False,  # BioBERT uses cased tokenization
                add_prefix_space=False
            )
            
            # Add medical terms if not already present
            self._add_biomedical_tokens()
            
        except Exception as e:
            print(f"Warning: Could not load BioBERT tokenizer: {e}")
            self.tokenizer = None
    
    def _add_biomedical_tokens(self) -> None:
        """Add biomedical-specific tokens to the tokenizer."""
        if self.tokenizer is None:
            return
            
        # Common biomedical abbreviations and terms
        biomedical_tokens = [
            # Medical abbreviations
            "q.d.", "b.i.d.", "t.i.d.", "q.i.d.", "p.r.n.", "stat",
            "i.v.", "i.m.", "s.c.", "p.o.", "h.s.", "a.c.", "p.c.",
            # Medical prefixes/suffixes  
            "cardio-", "neuro-", "hemato-", "gastro-", "nephro-",
            "-itis", "-emia", "-oma", "-pathy", "-ectomy", "-scopy",
            # Common medical terms
            "myocardial", "infarction", "hypertension", "diabetes",
            "tachycardia", "bradycardia", "pneumonia", "sepsis"
        ]
        
        # Add tokens that aren't already in vocabulary
        new_tokens = []
        for token in biomedical_tokens:
            if token not in self.tokenizer.get_vocab():
                new_tokens.append(token)
        
        if new_tokens:
            added = self.tokenizer.add_tokens(new_tokens)
            if added > 0:
                print(f"Added {added} biomedical tokens to BioBERT tokenizer")
    
    def _setup_biomedical_embeddings(self) -> None:
        """Set up biomedical-specific embedding handling."""
        # Store original embedding layer for potential weight conversion
        if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
            self.original_embeddings = self.model.embeddings.word_embeddings
            self.vocab_size = self.original_embeddings.num_embeddings
            self.embedding_dim = self.original_embeddings.embedding_dim
        elif hasattr(self.model, 'bert') and hasattr(self.model.bert.embeddings, 'word_embeddings'):
            self.original_embeddings = self.model.bert.embeddings.word_embeddings
            self.vocab_size = self.original_embeddings.num_embeddings
            self.embedding_dim = self.original_embeddings.embedding_dim
        else:
            self.original_embeddings = None
            self.vocab_size = None
            self.embedding_dim = None
    
    def _initialize_weights(self) -> None:
        """Initialize any additional weights or parameters."""
        # Handle weight conversion if tokenizer was extended
        if self.tokenizer is not None and self.original_embeddings is not None:
            current_vocab_size = len(self.tokenizer)
            if current_vocab_size > self.vocab_size:
                self._extend_embeddings(current_vocab_size)
    
    def _extend_embeddings(self, new_vocab_size: int) -> None:
        """Extend embedding layer for new biomedical tokens.
        
        Args:
            new_vocab_size: New vocabulary size after adding biomedical tokens
        """
        if self.original_embeddings is None:
            return
            
        # Create new embedding layer with extended vocabulary
        old_embeddings = self.original_embeddings
        new_embeddings = nn.Embedding(
            new_vocab_size, 
            self.embedding_dim,
            padding_idx=old_embeddings.padding_idx
        )
        
        # Copy existing weights
        with torch.no_grad():
            new_embeddings.weight[:self.vocab_size] = old_embeddings.weight
            
            # Initialize new token embeddings with mean of existing embeddings
            if new_vocab_size > self.vocab_size:
                mean_embedding = old_embeddings.weight.mean(dim=0)
                std_embedding = old_embeddings.weight.std(dim=0)
                for i in range(self.vocab_size, new_vocab_size):
                    # Initialize with slight variation around mean
                    new_embeddings.weight[i] = mean_embedding + torch.randn_like(mean_embedding) * std_embedding * 0.1
        
        # Replace the embedding layer in the model
        if hasattr(self.model, 'embeddings'):
            self.model.embeddings.word_embeddings = new_embeddings
        elif hasattr(self.model, 'bert'):
            self.model.bert.embeddings.word_embeddings = new_embeddings
            
        # Update stored references
        self.original_embeddings = new_embeddings
        self.vocab_size = new_vocab_size
        
        print(f"Extended BioBERT embeddings from {old_embeddings.num_embeddings} to {new_vocab_size} tokens")

    def setup_for_inference(self, use_cuda_graphs: bool = False, **kwargs) -> None:
        """Set up BioBERT for inference with optimizations.

        Args:
            use_cuda_graphs: Whether to enable CUDA graph optimization
            **kwargs: Additional optimization parameters
        """
        self.model.eval()

        # Initialize KV cache if not already done
        if self.kv_cache is None:
            self.kv_cache = self._initialize_kv_cache()

        # Set up CUDA graphs if requested and available
        if use_cuda_graphs and torch.cuda.is_available():
            self._setup_cuda_graphs()

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for BioBERT with optional KV caching.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            **kwargs: Additional arguments including:
                - attention_mask: Attention mask [batch_size, seq_len]
                - use_cache: Whether to use KV caching

        Returns:
            Model outputs [batch_size, seq_len, hidden_size]
        """
        attention_mask = kwargs.get("attention_mask")
        use_cache = kwargs.pop("use_cache", True)

        # Use KV cache if available and enabled
        if use_cache and self.kv_cache is not None:
            outputs = self._forward_with_cache(input_ids, attention_mask, **kwargs)
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)

        # Return logits for classification or sequence output for generation
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def _initialize_kv_cache(self) -> Dict[str, torch.Tensor]:
        """Initialize KV cache for efficient autoregressive generation.

        Returns:
            Dictionary containing initialized key and value caches
        """
        # Get model configuration
        batch_size = 1  # Default batch size
        max_seq_len = getattr(self.model.config, "max_position_embeddings", 512)

        # Initialize cache for each layer
        cache: Dict[str, torch.Tensor] = {}
        for layer_idx in range(self.num_hidden_layers):
            # Key cache: [batch_size, num_heads, max_seq_len, head_dim]
            key_cache = torch.zeros(
                batch_size,
                self.num_attention_heads,
                max_seq_len,
                self.head_dim,
                dtype=torch.float16,
                device="cpu",
            )
            # Value cache: [batch_size, num_heads, max_seq_len, head_dim]
            value_cache = torch.zeros(
                batch_size,
                self.num_attention_heads,
                max_seq_len,
                self.head_dim,
                dtype=torch.float16,
                device="cpu",
            )
            cache[f"layer_{layer_idx}_key"] = key_cache
            cache[f"layer_{layer_idx}_value"] = value_cache

        return cache

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with KV caching for efficient generation.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional model-specific arguments

        Returns:
            Model outputs
        """
        # Get current sequence length
        seq_len = input_ids.size(1)

        # Update attention mask for cached sequences
        if attention_mask is not None:
            attention_mask = self._update_attention_mask_for_cache(attention_mask)

        # Forward pass with past_key_values for caching
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=self._get_past_key_values(seq_len),
            **kwargs,
        )

        # Update KV cache with new keys and values
        self._update_kv_cache(outputs.past_key_values)

        return outputs

    def _update_attention_mask_for_cache(
        self, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Update attention mask to account for cached sequence.

        Args:
            attention_mask: Original attention mask

        Returns:
            Updated attention mask
        """
        # For simplicity, assume all cached tokens are not masked
        # In practice, you might want to track which positions are cached
        batch_size = attention_mask.size(0)
        cache_len = self._get_cache_sequence_length()

        # Create attention mask for cached positions (all ones)
        cache_mask = torch.ones(
            batch_size,
            cache_len,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        # Concatenate cache mask with input mask
        return torch.cat([cache_mask, attention_mask], dim=1)

    def _get_past_key_values(
        self, seq_len: int
    ) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """Get past key values for the current sequence position."""
        if self.kv_cache is None:
            raise RuntimeError("KV cache not initialized")

        past_key_values = []
        for i in range(self.num_hidden_layers):
            # Slice the cache to the current sequence length
            k = self.kv_cache[f"layer_{i}_key"][:, :, :seq_len, :]
            v = self.kv_cache[f"layer_{i}_value"][:, :, :seq_len, :]
            past_key_values.append((k, v))
        return tuple(past_key_values)

    def _update_kv_cache(
        self, new_key_values: Tuple[Tuple[torch.Tensor, ...], ...]
    ) -> None:
        """Update the KV cache with new key and value tensors."""
        if self.kv_cache is None:
            raise RuntimeError("KV cache not initialized")

        for i, (k, v) in enumerate(new_key_values):
            # Update cache with new keys and values
            # This assumes the cache is large enough to hold the new values
            self.kv_cache[f"layer_{i}_key"][:, :, : k.size(2), :] = k
            self.kv_cache[f"layer_{i}_value"][:, :, : v.size(2), :] = v

    def _get_cache_sequence_length(self) -> int:
        """Get the current sequence length in the cache."""
        if not self.kv_cache:
            return 0
        # All caches should have the same sequence length
        return next(iter(self.kv_cache.values())).size(2)

    def _setup_cuda_graphs(self) -> None:
        """Set up CUDA graphs for faster inference."""
        if not torch.cuda.is_available():
            return

        # Create CUDA stream and graph
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Warmup
            dummy_input = torch.zeros(1, 1, dtype=torch.long, device="cuda")
            _ = self(dummy_input)

            # Create graph
            self.cuda_graphs = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graphs):
                self._cuda_graph_input = dummy_input
                self._cuda_graph_output = self(dummy_input)
    
    def preprocess_biomedical_text(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Preprocess biomedical text with BioBERT-specific tokenization.
        
        Args:
            text: Input text(s) to preprocess
            **kwargs: Additional tokenization arguments
            
        Returns:
            Dictionary with tokenized inputs ready for the model
        """
        if self.tokenizer is None:
            raise RuntimeError("BioBERT tokenizer not initialized")
        
        # Set BioBERT-specific defaults
        defaults = {
            'max_length': 512,
            'padding': True,
            'truncation': True,
            'return_tensors': 'pt',
            'add_special_tokens': True
        }
        defaults.update(kwargs)
        
        # Handle biomedical text preprocessing
        if isinstance(text, str):
            # Preserve medical abbreviations and terms
            processed_text = self._preserve_medical_terms(text)
        else:
            processed_text = [self._preserve_medical_terms(t) for t in text]
        
        return self.tokenizer(processed_text, **defaults)
    
    def _preserve_medical_terms(self, text: str) -> str:
        """Preserve medical terms and abbreviations during tokenization.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text with preserved medical terms
        """
        # Add spaces around medical abbreviations to prevent splitting
        import re
        
        # Pattern for medical abbreviations (e.g., "q.d.", "b.i.d.")
        abbrev_pattern = r'\b([a-z]\.)+[a-z]?\.?\b'
        
        # Pattern for medical prefixes/suffixes
        prefix_pattern = r'\b(cardio|neuro|hemato|gastro|nephro|hepat)-'
        suffix_pattern = r'-(itis|emia|oma|pathy|plasty|ectomy|otomy|scopy)\b'
        
        # Preserve abbreviations
        text = re.sub(abbrev_pattern, r' \1 ', text, flags=re.IGNORECASE)
        
        # Preserve medical prefixes and suffixes
        text = re.sub(prefix_pattern, r' \1- ', text, flags=re.IGNORECASE)
        text = re.sub(suffix_pattern, r' -\1 ', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def convert_huggingface_weights(self, hf_model_path: str) -> None:
        """Convert Hugging Face BioBERT weights to Nano vLLM format.
        
        Args:
            hf_model_path: Path to Hugging Face BioBERT model
        """
        try:
            from transformers import AutoModel
            
            # Load the Hugging Face model
            hf_model = AutoModel.from_pretrained(hf_model_path)
            
            # Convert weights layer by layer
            self._convert_embedding_weights(hf_model)
            self._convert_encoder_weights(hf_model)
            self._convert_pooler_weights(hf_model)
            
            print(f"Successfully converted weights from {hf_model_path}")
            
        except Exception as e:
            print(f"Warning: Could not convert weights from {hf_model_path}: {e}")
    
    def _convert_embedding_weights(self, hf_model) -> None:
        """Convert embedding layer weights from Hugging Face format."""
        if not hasattr(hf_model, 'embeddings'):
            return
            
        hf_embeddings = hf_model.embeddings
        
        # Convert word embeddings
        if hasattr(self.model, 'embeddings') and hasattr(hf_embeddings, 'word_embeddings'):
            with torch.no_grad():
                self.model.embeddings.word_embeddings.weight.copy_(hf_embeddings.word_embeddings.weight)
        
        # Convert position embeddings
        if (hasattr(self.model.embeddings, 'position_embeddings') and 
            hasattr(hf_embeddings, 'position_embeddings')):
            with torch.no_grad():
                self.model.embeddings.position_embeddings.weight.copy_(hf_embeddings.position_embeddings.weight)
        
        # Convert token type embeddings
        if (hasattr(self.model.embeddings, 'token_type_embeddings') and 
            hasattr(hf_embeddings, 'token_type_embeddings')):
            with torch.no_grad():
                self.model.embeddings.token_type_embeddings.weight.copy_(hf_embeddings.token_type_embeddings.weight)
    
    def _convert_encoder_weights(self, hf_model) -> None:
        """Convert encoder layer weights from Hugging Face format."""
        if not hasattr(hf_model, 'encoder') or not hasattr(self.model, 'encoder'):
            return
            
        hf_layers = hf_model.encoder.layer
        model_layers = self.model.encoder.layer
        
        for i, (hf_layer, model_layer) in enumerate(zip(hf_layers, model_layers)):
            try:
                # Convert attention weights
                if hasattr(hf_layer, 'attention') and hasattr(model_layer, 'attention'):
                    self._convert_attention_weights(hf_layer.attention, model_layer.attention)
                
                # Convert feed-forward weights
                if hasattr(hf_layer, 'intermediate') and hasattr(model_layer, 'intermediate'):
                    with torch.no_grad():
                        model_layer.intermediate.dense.weight.copy_(hf_layer.intermediate.dense.weight)
                        model_layer.intermediate.dense.bias.copy_(hf_layer.intermediate.dense.bias)
                
                if hasattr(hf_layer, 'output') and hasattr(model_layer, 'output'):
                    with torch.no_grad():
                        model_layer.output.dense.weight.copy_(hf_layer.output.dense.weight)
                        model_layer.output.dense.bias.copy_(hf_layer.output.dense.bias)
                        
            except Exception as e:
                print(f"Warning: Could not convert layer {i} weights: {e}")
    
    def _convert_attention_weights(self, hf_attention, model_attention) -> None:
        """Convert attention weights from Hugging Face format."""
        try:
            with torch.no_grad():
                # Convert self-attention weights
                if hasattr(hf_attention.self, 'query') and hasattr(model_attention.self, 'query'):
                    model_attention.self.query.weight.copy_(hf_attention.self.query.weight)
                    model_attention.self.query.bias.copy_(hf_attention.self.query.bias)
                
                if hasattr(hf_attention.self, 'key') and hasattr(model_attention.self, 'key'):
                    model_attention.self.key.weight.copy_(hf_attention.self.key.weight)
                    model_attention.self.key.bias.copy_(hf_attention.self.key.bias)
                
                if hasattr(hf_attention.self, 'value') and hasattr(model_attention.self, 'value'):
                    model_attention.self.value.weight.copy_(hf_attention.self.value.weight)
                    model_attention.self.value.bias.copy_(hf_attention.self.value.bias)
                
                # Convert output projection weights
                if hasattr(hf_attention, 'output') and hasattr(model_attention, 'output'):
                    model_attention.output.dense.weight.copy_(hf_attention.output.dense.weight)
                    model_attention.output.dense.bias.copy_(hf_attention.output.dense.bias)
                    
        except Exception as e:
            print(f"Warning: Could not convert attention weights: {e}")
    
    def _convert_pooler_weights(self, hf_model) -> None:
        """Convert pooler weights from Hugging Face format."""
        if (hasattr(hf_model, 'pooler') and hasattr(self.model, 'pooler') and
            hasattr(hf_model.pooler, 'dense') and hasattr(self.model.pooler, 'dense')):
            try:
                with torch.no_grad():
                    self.model.pooler.dense.weight.copy_(hf_model.pooler.dense.weight)
                    self.model.pooler.dense.bias.copy_(hf_model.pooler.dense.bias)
            except Exception as e:
                print(f"Warning: Could not convert pooler weights: {e}")

    def to(self, device: torch.device) -> "BioBERTAdapter":
        """Move the model to the specified device."""
        super().to(device)
        # Move KV cache to the same device
        if self.kv_cache is not None:
            for k in self.kv_cache:
                self.kv_cache[k] = self.kv_cache[k].to(device)
        return self


class ClinicalBERTAdapter(MedicalModelAdapter):
    """Adapter for ClinicalBERT models."""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.model_type = "clinicalbert"

    def setup_for_inference(self, use_cuda_graphs: bool = False, **kwargs) -> None:
        """Set up ClinicalBERT for inference with optimizations."""
        self.model.eval()

        # Initialize KV cache if not already done
        if self.kv_cache is None:
            self.kv_cache = self._initialize_kv_cache()

        # Set up CUDA graphs if requested and available
        if use_cuda_graphs and torch.cuda.is_available():
            self._setup_cuda_graphs()

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for ClinicalBERT."""
        attention_mask = kwargs.get("attention_mask")

        # Use KV cache if available
        if self.kv_cache is not None:
            outputs = self._forward_with_cache(input_ids, attention_mask, **kwargs)
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)

        return outputs[0] if isinstance(outputs, tuple) else outputs

    def _initialize_kv_cache(self) -> Dict[str, torch.Tensor]:
        """Initialize KV cache for efficient autoregressive generation.

        Returns:
            Dictionary containing initialized key and value caches
        """
        # Get model configuration
        batch_size = 1  # Default batch size
        max_seq_len = getattr(self.model.config, "max_position_embeddings", 512)

        # Initialize cache for each layer
        cache: Dict[str, torch.Tensor] = {}
        for layer_idx in range(self.num_hidden_layers):
            # Key cache: [batch_size, num_heads, max_seq_len, head_dim]
            key_cache = torch.zeros(
                batch_size,
                self.num_attention_heads,
                max_seq_len,
                self.head_dim,
                dtype=torch.float16,
                device="cpu",
            )
            # Value cache: [batch_size, num_heads, max_seq_len, head_dim]
            value_cache = torch.zeros(
                batch_size,
                self.num_attention_heads,
                max_seq_len,
                self.head_dim,
                dtype=torch.float16,
                device="cpu",
            )
            cache[f"layer_{layer_idx}_key"] = key_cache
            cache[f"layer_{layer_idx}_value"] = value_cache

        return cache

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with KV caching for efficient generation.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional model-specific arguments

        Returns:
            Model outputs
        """
        # Get current sequence length
        seq_len = input_ids.size(1)

        # Update attention mask to include cache
        if attention_mask is not None:
            attention_mask = self._update_attention_mask_for_cache(attention_mask)

        # Forward pass with past_key_values for caching
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=self._get_past_key_values(seq_len),
            **kwargs,
        )

        # Update KV cache with new keys and values
        self._update_kv_cache(outputs.past_key_values)

        return outputs

    def _setup_cuda_graphs(self) -> None:
        """Set up CUDA graphs for faster inference."""
        if not torch.cuda.is_available():
            return

        # Create CUDA stream and graph
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Warmup
            dummy_input = torch.zeros(1, 1, dtype=torch.long, device="cuda")
            _ = self(dummy_input)

            # Create graph
            self.cuda_graphs = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graphs):
                self._cuda_graph_input = dummy_input
                self._cuda_graph_output = self(dummy_input)

    def to(self, device: torch.device) -> "ClinicalBERTAdapter":
        """Move adapter to specified device."""
        self.model = self.model.to(device)
        if self.kv_cache is not None:
            for k in self.kv_cache:
                self.kv_cache[k] = self.kv_cache[k].to(device)
        return self


def create_medical_adapter(
    model: nn.Module, model_type: str, config: Dict[str, Any]
) -> MedicalModelAdapter:
    """Factory function to create the appropriate adapter for a medical model.

    Args:
        model: The model to adapt
        model_type: Type of the model ('biobert' or 'clinicalbert')
        config: Configuration for the adapter

    Returns:
        An instance of the appropriate adapter
    """
    model_type = model_type.lower()
    if model_type == "biobert":
        return BioBERTAdapter(model, config)
    elif model_type == "clinicalbert":
        return ClinicalBERTAdapter(model, config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


__all__ = [
    "MedicalModelAdapter",
    "BioBERTAdapter",
    "ClinicalBERTAdapter",
    "create_medical_adapter",
]
