"""ClinicalBERT adapter implementation for clinical NLP tasks.

This module provides the ClinicalBERT adapter that handles clinical domain-specific
features, terminology, weight conversion, and contextual clinical note processing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Type, TYPE_CHECKING

import torch
import torch.nn as nn

# Lazy imports for transformers
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Define a dummy class for type checking
    class PreTrainedTokenizerBase:
        pass
    AutoTokenizer = None

from .base import MedicalModelAdapterBase


class ClinicalBERTAdapter(MedicalModelAdapterBase):
    """Adapter for ClinicalBERT models optimized for clinical NLP tasks.

    This adapter handles:
    - Clinical domain-specific tokenization and terminology
    - Weight conversion from Hugging Face format
    - Clinical note processing and contextual representations
    - CUDA graph optimization for clinical workflows
    """

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> 'ClinicalBERTAdapter':
        """Load a pre-trained ClinicalBERT model and return an adapter instance.
        
        Args:
            model_name_or_path: Name or path of the pre-trained model
            **kwargs: Additional arguments to pass to the model and adapter
            
        Returns:
            An instance of ClinicalBERTAdapter with the loaded model
            
        Example:
            >>> adapter = ClinicalBERTAdapter.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("The transformers library is required to load pre-trained models. "
                           "Please install it with: pip install transformers")
        
        from transformers import AutoModel
        
        # Load the model and tokenizer
        model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        
        # Create config dictionary for the adapter
        config = {
            "model_name_or_path": model_name_or_path,
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
            "use_cuda_graphs": kwargs.get("use_cuda_graphs", False),
            "memory_efficient": kwargs.get("memory_efficient", True),
            "enable_mixed_precision": kwargs.get("enable_mixed_precision", False),
        }
        
        # Create and return the adapter instance
        return cls(model=model, config=config)
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.model_type = "clinicalbert"
        self.num_hidden_layers = getattr(model.config, "num_hidden_layers", 12)
        self.num_attention_heads = getattr(model.config, "num_attention_heads", 12)
        self.hidden_size = getattr(model.config, "hidden_size", 768)
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Initialize tokenizer for ClinicalBERT-specific handling
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        if not config.get("skip_tokenizer_setup", False):
            self._setup_clinical_tokenizer()

        # Initialize weights and handle clinical embeddings
        self._initialize_weights()
        self._setup_clinical_embeddings()

    def _setup_clinical_tokenizer(self) -> None:
        """Set up ClinicalBERT-specific tokenizer with clinical vocabulary."""
        try:
            # Try to get tokenizer from model config or create default
            model_name = getattr(
                self.model.config, "_name_or_path", "emilyalsentzer/Bio_ClinicalBERT"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                do_lower_case=False,  # ClinicalBERT uses cased tokenization
                trust_remote_code=True,
            )

            # Add clinical tokens
            self._add_clinical_tokens()

        except Exception as e:
            print(f"Warning: Could not load ClinicalBERT tokenizer: {e}")
            self.tokenizer = None

    def _add_clinical_tokens(self) -> None:
        """Add clinical-specific tokens to the tokenizer."""
        clinical_tokens = [
            # Clinical abbreviations
            "COPD",
            "CHF",
            "MI",
            "CVA",
            "DVT",
            "PE",
            "UTI",
            "MRSA",
            "HTN",
            "DM",
            "CAD",
            "CABG",
            "PCI",
            "ICU",
            "ER",
            "OR",
            "CCU",
            "PACU",
            "NICU",
            "PICU",
            # Clinical procedures
            "intubation",
            "extubation",
            "tracheostomy",
            "thoracentesis",
            "paracentesis",
            "bronchoscopy",
            "endoscopy",
            "colonoscopy",
            "catheterization",
            "dialysis",
            # Clinical conditions
            "acute",
            "chronic",
            "exacerbation",
            "remission",
            "stable",
            "progressive",
            "deteriorating",
            "improving",
            "resolved",
            "persistent",
            "recurrent",
            # Clinical measurements
            "systolic",
            "diastolic",
            "tachycardic",
            "bradycardic",
            "hypertensive",
            "hypotensive",
            "febrile",
            "afebrile",
            # Clinical notes terminology
            "SOAP",
            "H&P",
            "discharge",
            "admission",
            "consult",
            "progress",
            "assessment",
            "plan",
            "impression",
            "differential",
        ]

        # Add tokens that aren't already in vocabulary
        new_tokens: List[str] = []
        # Add clinical tokens if they don't exist
        if self.tokenizer is not None:
            vocab = self.tokenizer.get_vocab()
            tokens_to_add = [t for t in clinical_tokens if t not in vocab]
            if tokens_to_add:
                added = self.tokenizer.add_tokens(tokens_to_add)
                print(f"Added {added} clinical tokens to ClinicalBERT tokenizer")

    def _setup_clinical_embeddings(self) -> None:
        """Set up clinical-specific embedding handling."""
        # Store original embedding layer for potential weight conversion
        if hasattr(self.model, "embeddings") and hasattr(
            self.model.embeddings, "word_embeddings"
        ):
            self.original_embeddings = self.model.embeddings.word_embeddings
            self.vocab_size = self.original_embeddings.num_embeddings
            self.embedding_dim = self.original_embeddings.embedding_dim
        elif hasattr(self.model, "bert") and hasattr(
            self.model.bert.embeddings, "word_embeddings"
        ):
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
        """Extend embedding layer for new clinical tokens."""
        if self.original_embeddings is None:
            return

        old_embeddings = self.original_embeddings

        # Create new embedding layer with extended vocabulary
        new_embeddings = nn.Embedding(new_vocab_size, self.embedding_dim)

        # Copy existing weights
        with torch.no_grad():
            new_embeddings.weight[: self.vocab_size] = old_embeddings.weight

            # Initialize new token embeddings with slight variation of existing embeddings
            if new_vocab_size > self.vocab_size:
                # Use mean and std of existing embeddings for initialization
                mean_embedding = old_embeddings.weight.mean(dim=0)
                std_embedding = old_embeddings.weight.std(dim=0)
                for i in range(self.vocab_size, new_vocab_size):
                    # Initialize with slight variation around mean
                    new_embeddings.weight[i] = (
                        mean_embedding
                        + torch.randn_like(mean_embedding) * std_embedding * 0.1
                    )

        # Replace the embedding layer in the model
        if hasattr(self.model, "embeddings"):
            self.model.embeddings.word_embeddings = new_embeddings
        elif hasattr(self.model, "bert"):
            self.model.bert.embeddings.word_embeddings = new_embeddings

        # Update stored references
        self.original_embeddings = new_embeddings
        self.vocab_size = new_vocab_size

        print(
            f"Extended ClinicalBERT embeddings from {old_embeddings.num_embeddings} to {new_vocab_size} tokens"
        )

    def setup_for_inference(self, use_cuda_graphs: bool = False, **kwargs) -> None:
        """Set up ClinicalBERT for inference with optimizations."""
        self.model.eval()

        # Update configuration from kwargs
        self.use_cuda_graphs = use_cuda_graphs or self.use_cuda_graphs
        self.memory_efficient = kwargs.get("memory_efficient", self.memory_efficient)
        self.enable_mixed_precision = kwargs.get(
            "enable_mixed_precision", self.enable_mixed_precision
        )

        # Initialize KV cache if not already done
        if self.kv_cache is None:
            self.kv_cache = self._initialize_kv_cache()

        # Setup tensor parallelism for model layers
        if self.tensor_parallel_size > 1:
            self._setup_tensor_parallel_layers()

        # Optimize CUDA memory usage
        self._optimize_cuda_memory()

        # Setup mixed precision if enabled
        self._setup_mixed_precision()

        # Set up CUDA graphs if requested and available
        if self.use_cuda_graphs and torch.cuda.is_available():
            self._setup_cuda_graphs()

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for ClinicalBERT."""
        # Use KV cache if available and requested
        use_cache = kwargs.get("use_cache", True)
        if use_cache and self.kv_cache is not None:
            # Implementation would use cached key-value pairs
            pass

        # Forward through the model
        with torch.no_grad():
            outputs = self.model(input_ids, **kwargs)

        return outputs

    def _initialize_kv_cache(self) -> Dict[str, torch.Tensor]:
        """Initialize KV cache for efficient inference."""
        cache = {}
        batch_size = 1  # Default batch size
        max_seq_len = 512  # Default max sequence length

        # Create cache for each layer
        for i in range(self.num_hidden_layers):
            # Key cache: [batch_size, num_heads, max_seq_len, head_dim]
            cache[f"layer_{i}_key"] = torch.zeros(
                batch_size,
                self.num_attention_heads,
                max_seq_len,
                self.head_dim,
                dtype=torch.float16 if self.enable_mixed_precision else torch.float32,
            )
            # Value cache: [batch_size, num_heads, max_seq_len, head_dim]
            cache[f"layer_{i}_value"] = torch.zeros(
                batch_size,
                self.num_attention_heads,
                max_seq_len,
                self.head_dim,
                dtype=torch.float16 if self.enable_mixed_precision else torch.float32,
            )

        return cache

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

    def _setup_tensor_parallel_layers(self) -> None:
        """Setup tensor parallelism for ClinicalBERT model layers."""
        if self.tensor_parallel_size <= 1:
            return

        print(
            f"Setting up tensor parallelism for ClinicalBERT (rank {self.rank}/{self.tensor_parallel_size})"
        )

        # Shard attention layers across tensor parallel ranks
        if hasattr(self.model, "bert") and hasattr(self.model.bert, "encoder"):
            for layer_idx, layer in enumerate(self.model.bert.encoder.layer):
                if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
                    attention = layer.attention.self

                    # Shard query, key, value projections
                    if hasattr(attention, "query"):
                        attention.query.weight.data = self._shard_tensor(
                            attention.query.weight.data, dim=0
                        )
                        if attention.query.bias is not None:
                            attention.query.bias.data = self._shard_tensor(
                                attention.query.bias.data, dim=0
                            )

                    if hasattr(attention, "key"):
                        attention.key.weight.data = self._shard_tensor(
                            attention.key.weight.data, dim=0
                        )
                        if attention.key.bias is not None:
                            attention.key.bias.data = self._shard_tensor(
                                attention.key.bias.data, dim=0
                            )

                    if hasattr(attention, "value"):
                        attention.value.weight.data = self._shard_tensor(
                            attention.value.weight.data, dim=0
                        )
                        if attention.value.bias is not None:
                            attention.value.bias.data = self._shard_tensor(
                                attention.value.bias.data, dim=0
                            )

                # Shard feed-forward layers
                if hasattr(layer, "intermediate") and hasattr(
                    layer.intermediate, "dense"
                ):
                    layer.intermediate.dense.weight.data = self._shard_tensor(
                        layer.intermediate.dense.weight.data, dim=0
                    )
                    if layer.intermediate.dense.bias is not None:
                        layer.intermediate.dense.bias.data = self._shard_tensor(
                            layer.intermediate.dense.bias.data, dim=0
                        )

                # Shard output projection (needs gathering)
                if hasattr(layer, "output") and hasattr(layer.output, "dense"):
                    layer.output.dense.weight.data = self._shard_tensor(
                        layer.output.dense.weight.data, dim=1
                    )

        # Shard embedding layers if needed
        if hasattr(self.model, "bert") and hasattr(self.model.bert, "embeddings"):
            embeddings = self.model.bert.embeddings
            if hasattr(embeddings, "word_embeddings"):
                embeddings.word_embeddings.weight.data = self._shard_tensor(
                    embeddings.word_embeddings.weight.data, dim=1
                )

        print(f"Tensor parallelism setup complete for ClinicalBERT (rank {self.rank})")

    def preprocess_clinical_text(
        self, text: Union[str, List[str]], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Preprocess clinical text with ClinicalBERT-specific tokenization.

        Args:
            text: Input clinical text(s) to preprocess
            **kwargs: Additional tokenization arguments

        Returns:
            Dictionary with tokenized inputs ready for the model
        """
        if self.tokenizer is None:
            raise RuntimeError("ClinicalBERT tokenizer not initialized")

        # Set ClinicalBERT-specific defaults
        defaults = {
            "max_length": 512,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            "add_special_tokens": True,
        }

        # Merge with user-provided kwargs
        tokenizer_kwargs = {**defaults, **kwargs}

        # Preserve clinical terms if text is string
        if isinstance(text, str):
            text = self._preserve_clinical_context(text)
        elif isinstance(text, list):
            text = [self._preserve_clinical_context(t) for t in text]

        # Tokenize the text
        encoded = self.tokenizer(text, **tokenizer_kwargs)

        return encoded

    def _preserve_clinical_context(self, text: str) -> str:
        """Preserve clinical terms and context during tokenization."""
        import re

        # Patterns for clinical terms that should be preserved
        clinical_patterns = [
            r"\b\d+\s*mg\b",  # Dosages like "5mg"
            r"\b\d+\s*ml\b",  # Volumes like "10ml"
            r"\b\d+\s*mcg\b",  # Micrograms
            r"\b\d+/\d+\b",  # Ratios like blood pressure "120/80"
            r"\b[A-Z]{2,}\b",  # Medical abbreviations like "COPD", "CHF"
            r"\b\w+(-\w+)+\b",  # Hyphenated clinical terms
            r"\b\d+\s*y/o\b",  # Age like "65 y/o"
            r"\bq\.d\.\b",  # Medical frequency abbreviations
            r"\bb\.i\.d\.\b",
            r"\bt\.i\.d\.\b",
            r"\bp\.r\.n\.\b",
        ]

        # Replace with placeholder tokens temporarily
        preserved_terms = {}
        for i, pattern in enumerate(clinical_patterns):
            matches = re.findall(pattern, text)
            for j, match in enumerate(matches):
                placeholder = f"__CLINICAL_TERM_{i}_{j}__"
                preserved_terms[placeholder] = match
                text = text.replace(match, placeholder, 1)

        return text

    def convert_huggingface_weights(self, hf_model_path: str) -> None:
        """Convert weights from Hugging Face ClinicalBERT format.

        Args:
            hf_model_path: Path to Hugging Face ClinicalBERT model
        """
        try:
            from transformers import AutoModel

            # Load the Hugging Face model
            hf_model = AutoModel.from_pretrained(hf_model_path)

            # Convert embedding weights
            self._convert_embedding_weights(hf_model)

            # Convert encoder weights
            self._convert_encoder_weights(hf_model)

            # Convert pooler weights if available
            self._convert_pooler_weights(hf_model)

            print(f"Successfully converted ClinicalBERT weights from {hf_model_path}")

        except Exception as e:
            print(f"Failed to convert ClinicalBERT weights: {e}")

    def _convert_embedding_weights(self, hf_model) -> None:
        """Convert embedding layer weights from Hugging Face format."""
        if hasattr(hf_model, "embeddings") and hasattr(self.model, "embeddings"):
            try:
                with torch.no_grad():
                    # Word embeddings
                    self.model.embeddings.word_embeddings.weight.copy_(
                        hf_model.embeddings.word_embeddings.weight
                    )
                    # Position embeddings
                    if hasattr(hf_model.embeddings, "position_embeddings"):
                        self.model.embeddings.position_embeddings.weight.copy_(
                            hf_model.embeddings.position_embeddings.weight
                        )
                    # Token type embeddings
                    if hasattr(hf_model.embeddings, "token_type_embeddings"):
                        self.model.embeddings.token_type_embeddings.weight.copy_(
                            hf_model.embeddings.token_type_embeddings.weight
                        )
            except Exception as e:
                print(f"Warning: Could not convert ClinicalBERT embedding weights: {e}")

    def _convert_encoder_weights(self, hf_model) -> None:
        """Convert encoder layer weights from Hugging Face format."""
        if not (hasattr(hf_model, "encoder") and hasattr(self.model, "encoder")):
            return

        hf_layers = hf_model.encoder.layer
        model_layers = self.model.encoder.layer

        for i, (hf_layer, model_layer) in enumerate(zip(hf_layers, model_layers)):
            try:
                # Convert attention weights
                if hasattr(hf_layer, "attention") and hasattr(model_layer, "attention"):
                    self._convert_attention_weights(
                        hf_layer.attention, model_layer.attention
                    )

                # Convert feed-forward weights
                if hasattr(hf_layer, "intermediate") and hasattr(
                    model_layer, "intermediate"
                ):
                    with torch.no_grad():
                        model_layer.intermediate.dense.weight.copy_(
                            hf_layer.intermediate.dense.weight
                        )
                        model_layer.intermediate.dense.bias.copy_(
                            hf_layer.intermediate.dense.bias
                        )

                if hasattr(hf_layer, "output") and hasattr(model_layer, "output"):
                    with torch.no_grad():
                        model_layer.output.dense.weight.copy_(
                            hf_layer.output.dense.weight
                        )
                        model_layer.output.dense.bias.copy_(hf_layer.output.dense.bias)

            except Exception as e:
                print(f"Warning: Could not convert layer {i} weights: {e}")

    def _convert_attention_weights(self, hf_attention, model_attention) -> None:
        """Convert attention weights from Hugging Face format."""
        try:
            with torch.no_grad():
                # Convert self-attention weights
                if hasattr(hf_attention.self, "query") and hasattr(
                    model_attention.self, "query"
                ):
                    model_attention.self.query.weight.copy_(
                        hf_attention.self.query.weight
                    )
                    model_attention.self.query.bias.copy_(hf_attention.self.query.bias)

                if hasattr(hf_attention.self, "key") and hasattr(
                    model_attention.self, "key"
                ):
                    model_attention.self.key.weight.copy_(hf_attention.self.key.weight)
                    model_attention.self.key.bias.copy_(hf_attention.self.key.bias)

                if hasattr(hf_attention.self, "value") and hasattr(
                    model_attention.self, "value"
                ):
                    model_attention.self.value.weight.copy_(
                        hf_attention.self.value.weight
                    )
                    model_attention.self.value.bias.copy_(hf_attention.self.value.bias)

                # Convert output projection weights
                if hasattr(hf_attention, "output") and hasattr(
                    model_attention, "output"
                ):
                    model_attention.output.dense.weight.copy_(
                        hf_attention.output.dense.weight
                    )
                    model_attention.output.dense.bias.copy_(
                        hf_attention.output.dense.bias
                    )

        except Exception as e:
            print(f"Warning: Could not convert ClinicalBERT attention weights: {e}")

    def _convert_pooler_weights(self, hf_model) -> None:
        """Convert pooler weights from Hugging Face format."""
        if (
            hasattr(hf_model, "pooler")
            and hasattr(self.model, "pooler")
            and hasattr(hf_model.pooler, "dense")
            and hasattr(self.model.pooler, "dense")
        ):
            try:
                with torch.no_grad():
                    self.model.pooler.dense.weight.copy_(hf_model.pooler.dense.weight)
                    self.model.pooler.dense.bias.copy_(hf_model.pooler.dense.bias)
            except Exception as e:
                print(f"Warning: Could not convert ClinicalBERT pooler weights: {e}")

    def process_clinical_note(
        self, note_text: str, note_type: str = "progress"
    ) -> Dict[str, torch.Tensor]:
        """Process clinical notes with contextual understanding.

        Args:
            note_text: Clinical note text
            note_type: Type of note (progress, discharge, admission, etc.)

        Returns:
            Processed clinical note ready for model input
        """
        # Add note type context
        if note_type.lower() in ["progress", "soap"]:
            context_prefix = "[PROGRESS NOTE] "
        elif note_type.lower() in ["discharge", "summary"]:
            context_prefix = "[DISCHARGE SUMMARY] "
        elif note_type.lower() in ["admission", "h&p"]:
            context_prefix = "[ADMISSION NOTE] "
        elif note_type.lower() == "consult":
            context_prefix = "[CONSULTATION] "
        else:
            context_prefix = "[CLINICAL NOTE] "

        # Preprocess with clinical context
        contextualized_text = context_prefix + note_text
        return self.preprocess_clinical_text(contextualized_text)

    def to(self, *args, **kwargs) -> "ClinicalBERTAdapter":
        """Move the adapter to the specified device and/or dtype.

        This method extends the base class to() method to handle the adapter's internal state.

        Args:
            *args: Arguments to pass to the parent to() method.
            **kwargs: Keyword arguments to pass to the parent to() method.

        Returns:
            self: The adapter moved to the specified device/dtype.
        """
        self.model = self.model.to(*args, **kwargs)
        if hasattr(self, "kv_cache") and self.kv_cache is not None:
            self.kv_cache = {k: v.to(*args, **kwargs) for k, v in self.kv_cache.items()}
        return self
