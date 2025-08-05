"""ClinicalBERT adapter with optimized attention and layer implementations."""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ...models.utils.attention_utils import apply_attention, combine_heads, split_heads
from ...models.utils.layer_utils import create_initializer, get_activation_fn
from ..attention import MedicalMultiheadAttention
from ..layers import MedicalFeedForward, MedicalLayerNorm
from .medical_adapter_base import MedicalModelAdapterBase


class ClinicalBERTAdapter(MedicalModelAdapterBase):
    """ClinicalBERT adapter with optimized attention and layer implementations.

    This adapter provides specialized optimizations for ClinicalBERT models,
    including clinical text tokenization, attention mechanisms, and layer structures.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the ClinicalBERT adapter.

        Args:
            model: Optional pre-initialized model
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
        """
        # Default config if not provided
        if config is None:
            config = {
                "vocab_size": 28996,  # ClinicalBERT vocab size
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
                "model_type": "bert",
            }

        # Initialize model if not provided
        if model is None:
            model = AutoModel.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT", config=config
            )

        # Initialize the base class with model first, then config
        super().__init__(model=model, config=config, **kwargs)
        
        # Set model type for identification
        self.model_type = "clinicalbert"
        
        # Set use_cuda based on device
        self.use_cuda = str(self.device).startswith('cuda')

        # Initialize tokenizer
        self.tokenizer = self._init_tokenizer()

        # Add clinical tokens
        self._add_clinical_tokens()

        # Initialize CUDA optimizations
        self._init_cuda_optimizations()

    def _init_tokenizer(self):
        """Initialize the ClinicalBERT tokenizer with clinical vocabulary."""
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        return tokenizer

    def _add_clinical_tokens(self) -> None:
        """Add clinical-specific tokens to the tokenizer."""
        clinical_tokens = [
            # Common clinical abbreviations
            "SOAP",
            "HPI",
            "ROS",
            "PMH",
            "PSH",
            "FH",
            "SH",
            "Meds",
            "Allergies",
            "CC",
            "EDC",
            "LMP",
            "NKA",
            "NKDA",
            "FHx",
            "PMHx",
            "PSHx",
            "SHx",
            "VS",
            "T",
            "P",
            "R",
            "BP",
            "HR",
            "RR",
            "O2",
            "SpO2",
            "Temp",
            "Wt",
            "Ht",
            "BMI",
            "BS",
            "I/O",
            "UOP",
            "EKG",
            "CXR",
            "CT",
            "MRI",
            "US",
            "XR",
            "CBC",
            "CMP",
            "BMP",
            "LFTs",
            "PT",
            "PTT",
            "INR",
            "Trop",
            "BNP",
            "ABG",
            # Common clinical terms
            "asymptomatic",
            "symptomatic",
            "afebrile",
            "febrile",
            "acute",
            "chronic",
            "stable",
            "unstable",
            "improving",
            "worsening",
            "resolved",
            "persistent",
            "mild",
            "moderate",
            "severe",
            "diffuse",
            "focal",
            "bilateral",
            "unilateral",
            "proximal",
            "distal",
            "superior",
            "inferior",
            "anterior",
            "posterior",
            "lateral",
            "medial",
            "superficial",
            "deep",
            "tender",
            "non-tender",
            # Common clinical findings
            "rales",
            "rhonchi",
            "wheezes",
            "crackles",
            "stridor",
            "absent",
            "decreased",
            "increased",
            "normal",
            "abnormal",
            "regular",
            "irregular",
            "bounding",
            "thready",
            "weak",
            "strong",
            "rapid",
            "slow",
            "shallow",
            "labored",
            # Common assessments and plans
            "admit",
            "discharge",
            "transfer",
            "consult",
            "followup",
            "f/u",
            "prn",
            "as needed",
            "continue",
            "discontinue",
            "increase",
            "decrease",
            "change",
            "start",
            "stop",
            "hold",
            "resume",
            "monitor",
            "observe",
            "consider",
            "rule out",
            "r/o",
            "versus",
            "vs",
            "versus",
            "versus",
        ]

        # Add tokens that aren't already in the tokenizer
        added_tokens = []
        for token in clinical_tokens:
            if token not in self.tokenizer.get_vocab():
                added_tokens.append(token)

        if added_tokens and hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.add_tokens(added_tokens)
            # Resize token embeddings to match new vocab size
            if (
                hasattr(self, "model")
                and self.model is not None
                and hasattr(self.model, "resize_token_embeddings")
            ):
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

    def preprocess_clinical_text(
        self, text: str, note_type: str = "progress"
    ) -> Dict[str, torch.Tensor]:
        """Preprocess clinical text for the model.

        Args:
            text: Input text to preprocess
            note_type: Type of clinical note (e.g., "progress", "discharge", "admission")

        Returns:
            Dictionary containing preprocessed inputs
        """
        # Clean and normalize the text
        text = self._clean_clinical_text(text, note_type)

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

    def _clean_clinical_text(self, text: str, note_type: str) -> str:
        """Clean and normalize clinical text.

        Args:
            text: Input text to clean
            note_type: Type of clinical note

        Returns:
            Cleaned text
        """
        # Add note type context
        if note_type.lower() == "progress":
            text = f"[PROGRESS NOTE] {text}"
        elif note_type.lower() == "discharge":
            text = f"[DISCHARGE SUMMARY] {text}"
        elif note_type.lower() == "admission":
            text = f"[ADMISSION NOTE] {text}"
        elif note_type.lower() == "consult":
            text = f"[CONSULT NOTE] {text}"

        # Preserve common clinical patterns
        text = re.sub(
            r"(\d+)([mM]?[gG]\b|mg/kg)", r"\1 \2", text
        )  # Add space before units
        text = re.sub(r"(\d+)([mM][mM][Hh][Gg]\b)", r"\1 \2", text)  # Handle mmHg
        text = re.sub(r"(\d+)([bB][pP][mM]\b)", r"\1 \2", text)  # Handle bpm
        text = re.sub(r"(\d+)([mM][gG]/[dD][lL]\b)", r"\1 \2", text)  # Handle mg/dL

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
        if not hasattr(self.model, "embeddings") or not hasattr(
            self.model.embeddings, "word_embeddings"
        ):
            raise AttributeError("Model does not have expected embeddings structure")
        return self.model.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the input embeddings."""
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been initialized")
        if not hasattr(self.model, "embeddings") or not hasattr(
            self.model.embeddings, "word_embeddings"
        ):
            raise AttributeError("Model does not have expected embeddings structure")
        self.model.embeddings.word_embeddings = value

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

        # Save model
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
        cls,
        pretrained_model_name_or_path: str = "emilyalsentzer/Bio_ClinicalBERT",
        **kwargs,
    ) -> "ClinicalBERTAdapter":
        """Load a pretrained ClinicalBERT model.

        Args:
            pretrained_model_name_or_path: Name or path of the pretrained model
            **kwargs: Additional keyword arguments

        Returns:
            ClinicalBERTAdapter instance
        """
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model=model)
