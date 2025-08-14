"""Medical Named Entity Recognition task implementation."""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from medvllm.tasks.base import MedicalTask
from medvllm.tasks.types import NERPredictionType


class MedicalNER(MedicalTask, nn.Module):
    """Medical Named Entity Recognition model."""

    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[Union[PreTrainedTokenizerBase, str]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        num_labels: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the Medical NER model.

        Args:
            model_name: Pretrained model name or path
            tokenizer: Tokenizer for the model
            tokenizer_kwargs: Additional arguments for the tokenizer
            num_labels: Number of entity types to predict
            **kwargs: Additional arguments for the model
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **(tokenizer_kwargs or {}))
        elif tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **(tokenizer_kwargs or {}))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.num_labels = num_labels

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights for the classifier layer."""
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for NER.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            labels: Ground truth labels for training

        Returns:
            Dictionary with logits and loss (if labels provided)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"logits": logits, "loss": loss, "hidden_states": outputs.hidden_states}

    def predict(
        self,
        texts: List[str],
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> List[NERPredictionType]:
        """Predict named entities in medical texts.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            device: Device to run inference on

        Returns:
            List of NER predictions for each input text
        """
        self.eval()
        self.to(device)

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self(**inputs)

        # Process predictions
        predictions = []
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        for i, (text, pred) in enumerate(zip(texts, preds)):
            # Convert token predictions to entities
            entities = []
            scores = []
            offsets = []

            # Get word tokens and their positions
            if not hasattr(self.tokenizer, "tokenize") or not callable(self.tokenizer.tokenize):
                raise ValueError("Tokenizer must have a callable 'tokenize' method")
            tokens = self.tokenizer.tokenize(text)
            word_ids = inputs.word_ids(batch_index=i)

            current_entity = None
            current_start = None
            current_score = 0.0

            for j, (token, word_id) in enumerate(zip(tokens, word_ids)):
                if word_id is None:  # Skip special tokens
                    continue

                label = pred[j]
                score = torch.softmax(logits[i, j], dim=-1)[label].item()

                # Simple IOB decoding
                if label % 2 == 1:  # B- or I- prefix
                    entity_type = label // 2

                    if current_entity is not None and current_entity == entity_type:
                        # Continue current entity
                        current_score = (current_score + score) / 2
                    else:
                        # Save previous entity if exists
                        if current_entity is not None:
                            entities.append(f"ENT_{current_entity}")
                            scores.append(current_score)
                            offsets.append((current_start, j - 1))

                        # Start new entity
                        current_entity = entity_type
                        current_start = j
                        current_score = score

            # Add last entity if exists
            if current_entity is not None:
                entities.append(f"ENT_{current_entity}")
                scores.append(current_score)
                offsets.append((current_start, len(tokens) - 1))

            predictions.append(
                NERPredictionType(
                    text=text,
                    entities=entities,
                    scores=[float(score) for score in scores],
                    offsets=[(s, e) for s, e in offsets if s is not None],
                )
            )

        return predictions
