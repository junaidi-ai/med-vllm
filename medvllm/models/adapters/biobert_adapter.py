"""BioBERT adapter with optimized attention and layer implementations."""

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

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

        if added_tokens:
            self.tokenizer.add_tokens(added_tokens)
            # Resize token embeddings to match new vocab size
            if hasattr(self, "model"):
                self.model.resize_token_embeddings(len(self.tokenizer))

    def _init_cuda_optimizations(self) -> None:
        """Initialize CUDA optimizations if available."""
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = torch.compile(self.model)  # Enable PyTorch 2.0 compilation

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

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Forward pass through the model."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the input embeddings."""
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        """Get the output embeddings."""
        return self.model.get_output_embeddings()

    def tie_weights(self) -> None:
        """Tie the weights between the input and output embeddings."""
        self.model.tie_weights()

    def save_pretrained(self, save_directory: str) -> None:
        """Save the model to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
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
