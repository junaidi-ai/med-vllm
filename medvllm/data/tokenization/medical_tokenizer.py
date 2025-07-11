"""Medical text tokenization with special handling for medical terminology."""

from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Common medical abbreviations and terms to add to the tokenizer
MEDICAL_TERMS = [
    # Medical abbreviations
    "q.d.",
    "b.i.d.",
    "t.i.d.",
    "q.i.d.",
    "q.h.",
    "q.4h.",
    "q.6h.",
    "q.8h.",
    "q.12h.",
    "q.d.",
    "q.o.d.",
    "q.w.",
    "q.m.",
    "q.a.m.",
    "q.p.m.",
    "a.c.",
    "p.c.",
    "p.r.n.",
    "stat",
    "h.s.",
    "p.o.",
    "i.v.",
    "i.m.",
    "s.c.",
    "s.l.",
    "p.r.",
    "p.v.",
    "o.d.",
    "o.s.",
    "o.u.",
    "a.d.",
    "a.s.",
    "a.u.",
    "ad lib",
    "c_",
    "s_",
    "NOS",
    "NOS.",
    # Common medical prefixes/suffixes
    "hemato-",
    "cardio-",
    "neuro-",
    "pulmono-",
    "gastro-",
    "nephro-",
    "hepat-",
    "-itis",
    "-emia",
    "-oma",
    "-pathy",
    "-plasty",
    "-ectomy",
    "-otomy",
    "-scopy",
    # Common medical terms
    "myocardial",
    "infarction",
    "hypertension",
    "diabetes",
    "mellitus",
    "tachycardia",
    "bradycardia",
    "arrhythmia",
    "pneumonia",
    "bronchitis",
    "asthma",
    "emphysema",
    "edema",
    "thrombosis",
    "embolism",
    "sepsis",
    "anemia",
    "leukemia",
    "lymphoma",
]


class MedicalTokenizer:
    """Wrapper around a tokenizer with medical term handling."""

    def __init__(
        self, tokenizer_name: str = "gpt2", add_medical_terms: bool = True, **kwargs
    ):
        """Initialize the medical tokenizer.

        Args:
            tokenizer_name: Name or path of the base tokenizer
            add_medical_terms: Whether to add medical terms to the tokenizer
            **kwargs: Additional arguments to pass to the tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
        self.medical_terms_added = False

        if add_medical_terms:
            self._add_medical_terms()

    def _add_medical_terms(self) -> None:
        """Add medical terms to the tokenizer's vocabulary."""
        if self.medical_terms_added:
            return

        # Get current vocabulary size
        current_vocab_size = len(self.tokenizer)

        # Add medical terms
        added = self.tokenizer.add_tokens(MEDICAL_TERMS)

        if added > 0:
            print(f"Added {added} medical terms to the tokenizer")

        self.medical_terms_added = True

    def __call__(self, *args, **kwargs):
        """Call the underlying tokenizer."""
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying tokenizer."""
        return getattr(self.tokenizer, name)

    def encode_medical_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: Optional[str] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict, List[int]]:
        """Encode medical text with special handling.

        Args:
            text: Input text to encode
            max_length: Maximum length of the encoding
            padding: Padding strategy ('max_length', 'longest', etc.)
            truncation: Whether to truncate to max_length
            return_tensors: Type of tensors to return ('pt', 'tf', 'np')
            **kwargs: Additional arguments to pass to the tokenizer

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' or list of token IDs
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            **kwargs,
        )

    def batch_encode_medical_text(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: Optional[str] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of medical texts.

        Args:
            texts: List of input texts to encode
            max_length: Maximum length of the encodings
            padding: Padding strategy ('max_length', 'longest', etc.)
            truncation: Whether to truncate to max_length
            return_tensors: Type of tensors to return ('pt', 'tf', 'np')
            **kwargs: Additional arguments to pass to the tokenizer

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding or "longest",
            truncation=truncation,
            return_tensors=return_tensors or "pt",
            **kwargs,
        )
