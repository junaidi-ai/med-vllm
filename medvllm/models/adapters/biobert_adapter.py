"""BioBERT adapter with optimized attention and layer implementations."""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from transformers import BioGptModel, BioGptTokenizer

from ...utils.attention_utils import apply_attention, combine_heads, split_heads
from ...utils.layer_utils import create_initializer, get_activation_fn
from ..attention import MedicalMultiheadAttention
from ..layers import MedicalFeedForward, MedicalLayerNorm
from .medical_adapter_base import MedicalModelAdapterBase


class BioBERTAdapter(MedicalModelAdapterBase):
    """BioBERT adapter with optimized attention and layer implementations.

    This adapter provides specialized optimizations for BioBERT models,
    including biomedical tokenization, attention mechanisms, and layer structures.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the BioBERT adapter.

        Args:
            model: Optional pre-initialized BioGptModel
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
        """
        # Default config if not provided
        if config is None:
            config = {
                "vocab_size": 42384,  # BioBERT vocab size
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 512,
                "type_vocab_size": 2,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }

        # Initialize the base class
        super().__init__(config=config, model=model, **kwargs)

        # Initialize tokenizer
        self.tokenizer = self._init_tokenizer()

        # Add biomedical tokens
        self._add_biomedical_tokens()

        # Initialize model if not provided
        if model is None:
            self.model = BioGptModel.from_pretrained(
                "microsoft/biogpt", config=self.config
            )

        # Initialize CUDA optimizations
        self._init_cuda_optimizations()

    def _init_tokenizer(self):
        """Initialize the BioGPT tokenizer with biomedical vocabulary."""
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        return tokenizer

    def _add_biomedical_tokens(self) -> None:
        """Add biomedical-specific tokens to the tokenizer."""
        biomedical_tokens = [
            # Medical abbreviations
            "q.d.",
            "b.i.d.",
            "t.i.d.",
            "q.i.d.",
            "p.r.n.",
            "a.c.",
            "p.c.",
            "h.s.",
            "stat",
            "ad lib",
            "NPO",
            "PRN",
            "QD",
            "BID",
            "TID",
            "QID",
            "AC",
            "PC",
            "HS",
            "PO",
            "IV",
            "IM",
            "SC",
            "SL",
            "PR",
            "PV",
            "NG",
            "OG",
            "ETOH",
            # Medical prefixes/suffixes
            "cardio-",
            "neuro-",
            "pulmo-",
            "hepato-",
            "nephro-",
            "osteo-",
            "arthro-",
            "-itis",
            "-emia",
            "-oma",
            "-pathy",
            "-plasty",
            "-ectomy",
            "-scopy",
            # Common medical terms
            "myocardial",
            "infarction",
            "hypertension",
            "diabetes",
            "pneumonia",
            "tachycardia",
            "bradycardia",
            "tachypnea",
            "dyspnea",
            "hypoxia",
            # Lab values
            "WBC",
            "RBC",
            "HGB",
            "HCT",
            "PLT",
            "NA",
            "K",
            "CL",
            "CO2",
            "BUN",
            "CR",
            "GLU",
            "CA",
            "MG",
            "PHOS",
            "AST",
            "ALT",
            "ALP",
            "TBIL",
            "ALB",
        ]

        # Add tokens that aren't already in the tokenizer
        added_tokens = []
        for token in biomedical_tokens:
            if token not in self.tokenizer.get_vocab():
                added_tokens.append(token)

        if added_tokens and hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.add_tokens(added_tokens)
            # Resize token embeddings to match new vocab size
            if hasattr(self, "model") and self.model is not None:
                if hasattr(self.model, "resize_token_embeddings"):
                    self.model.resize_token_embeddings(len(self.tokenizer))

    def _init_cuda_optimizations(self) -> None:
        """Initialize CUDA optimizations if available."""
        if (
            torch.cuda.is_available()
            and hasattr(self, "model")
            and self.model is not None
        ):
            try:
                self.model = self.model.cuda()
                # Only compile if model is on CUDA
                if self.model.device.type == "cuda":
                    self.model = torch.compile(
                        self.model
                    )  # Enable PyTorch 2.0 compilation
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA optimizations: {e}")

    def preprocess_biomedical_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess biomedical text for the model.

        Args:
            text: Input text to preprocess

        Returns:
            Dictionary containing preprocessed inputs
        """
        # Clean and normalize the text
        text = self._clean_biomedical_text(text)

        # Tokenize the text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.get("max_position_embeddings", 512)
            - 2,  # Account for [CLS] and [SEP]
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        # Move to device if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        return inputs

    def _clean_biomedical_text(self, text: str) -> str:
        """Clean and normalize biomedical text.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        # Preserve common medical patterns
        text = re.sub(
            r"(\d+)([mM]?[gG]\b|mg/kg)", r"\1 \2", text
        )  # Add space before units
        text = re.sub(r"(\d+)([mM][mM][Hh][Gg]\b)", r"\1 \2", text)  # Handle mmHg
        text = re.sub(r"(\d+)([bB][pP][mM]\b)", r"\1 \2", text)  # Handle bpm

        # Normalize whitespace
        text = " ".join(text.split())
        return text

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
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            token_type_ids: Token type IDs of shape (batch_size, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len)
            head_mask: Mask for attention heads
            inputs_embeds: Input embeddings instead of input_ids
            output_attentions: Whether to return attentions
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary containing model outputs
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been initialized")

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings."""
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been initialized")
        if not hasattr(self.model, "get_input_embeddings"):
            raise NotImplementedError("Model does not implement get_input_embeddings")
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the input embeddings."""
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been initialized")
        if not hasattr(self.model, "set_input_embeddings"):
            raise NotImplementedError("Model does not implement set_input_embeddings")
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        """Get the output embeddings."""
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been initialized")
        if not hasattr(self.model, "get_output_embeddings"):
            raise NotImplementedError("Model does not implement get_output_embeddings")
        embeddings = self.model.get_output_embeddings()
        if embeddings is None:
            raise ValueError("Model returned None for output embeddings")
        return embeddings

    def tie_weights(self) -> None:
        """Tie the weights between the input and output embeddings."""
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been initialized")
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()
        elif hasattr(self.model, "tie_word_embeddings"):
            self.model.tie_word_embeddings()

    def save_pretrained(self, save_directory: str) -> None:
        """Save the model to a directory."""
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been initialized")
        os.makedirs(save_directory, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_directory)
        else:
            torch.save(
                self.model.state_dict(),
                os.path.join(save_directory, "pytorch_model.bin"),
            )

        # Save tokenizer if available
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str = "microsoft/biogpt", **kwargs
    ) -> "BioBERTAdapter":
        """Load a pretrained BioBERT model.

        Args:
            pretrained_model_name_or_path: Name or path of the pretrained model
            **kwargs: Additional keyword arguments

        Returns:
            BioBERTAdapter instance
        """
        model = BioGptModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model=model)
