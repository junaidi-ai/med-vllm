"""Base classes for medical NLP tasks."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class MedicalTask(nn.Module, ABC):
    """Base class for all medical NLP tasks.

    This class provides a common interface and shared functionality for all medical
    NLP tasks in the Med vLLM framework.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the medical task.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass for the model.

        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method.")

    def save_pretrained(self, save_directory: str) -> None:
        """Save the model and tokenizer to a directory.

        Args:
            save_directory: Directory to save the model and tokenizer to.
        """
        if self.model is not None:
            self.model.save_pretrained(save_directory)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *args: Any, **kwargs: Any) -> "MedicalTask":
        """Load a pretrained model from a directory or model hub.

        Args:
            model_name_or_path: Name or path of the pretrained model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of the MedicalTask subclass.
        """
        raise NotImplementedError("Subclasses must implement from_pretrained method.")

    def to(self, device: Union[str, torch.device]) -> "MedicalTask":
        """Move the model to the specified device.

        Args:
            device: The device to move the model to.

        Returns:
            self: Returns the model.
        """
        super().to(device)
        if self.model is not None:
            self.model = self.model.to(device)
        return self

    def train_mode(self) -> None:
        """Set the model to training mode."""
        self.train()
        if self.model is not None:
            self.model.train()

    def eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        self.eval()
        if self.model is not None:
            self.model.eval()
