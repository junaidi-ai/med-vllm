"""Medical Question Answering task implementation."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


@dataclass
class QAPrediction:
    """Container for QA prediction results."""

    answer: str
    score: float
    start: int
    end: int
    context: str


class MedicalQA(nn.Module):
    """Medical Question Answering model."""

    def __init__(self, model_name: str = "bert-base-uncased", **kwargs):
        """Initialize the Medical QA model.

        Args:
            model_name: Pretrained model name or path
            **kwargs: Additional arguments for the model
        """
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for QA.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            start_positions: Start positions for training
            end_positions: End positions for training

        Returns:
            Dictionary with start/end logits and loss
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            **kwargs,
        )

        return {
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "loss": outputs.loss,
        }

    def predict(
        self,
        questions: List[str],
        contexts: List[str],
        max_length: int = 512,
        max_answer_length: int = 100,
        n_best: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> List[QAPrediction]:
        """Generate answers to medical questions.

        Args:
            questions: List of questions
            contexts: List of context passages
            max_length: Maximum sequence length
            max_answer_length: Maximum length of predicted answers
            n_best: Number of top predictions to return
            device: Device to run inference on

        Returns:
            List of QA predictions
        """
        self.eval()
        self.to(device)

        # Tokenize inputs
        inputs = self.tokenizer(
            questions,
            contexts,
            padding=True,
            truncation="only_second",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self(**inputs)

        # Process predictions
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]

        predictions = []

        for i, (question, context) in enumerate(zip(questions, contexts)):
            start_scores = start_logits[i].cpu().numpy()
            end_scores = end_logits[i].cpu().numpy()

            # Get the n-best predictions
            start_indices = start_scores.argsort()[-n_best:][::-1]
            end_indices = end_scores.argsort()[-n_best:][::-1]

            best_score = -1
            best_pred = None

            for start_idx in start_indices:
                for end_idx in end_indices:
                    # Skip invalid predictions
                    if end_idx < start_idx:
                        continue
                    if end_idx - start_idx + 1 > max_answer_length:
                        continue

                    score = start_scores[start_idx] + end_scores[end_idx]

                    if score > best_score:
                        best_score = score
                        best_pred = {"start": start_idx, "end": end_idx, "score": score}

            if best_pred is not None:
                # Convert token indices to text
                answer_tokens = inputs["input_ids"][i][best_pred["start"] : best_pred["end"] + 1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

                predictions.append(
                    QAPrediction(
                        answer=answer,
                        score=float(best_pred["score"]),
                        start=int(best_pred["start"]),
                        end=int(best_pred["end"]),
                        context=context,
                    )
                )
            else:
                predictions.append(
                    QAPrediction(answer="", score=0.0, start=0, end=0, context=context)
                )

        return predictions
