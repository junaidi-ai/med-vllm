"""Type definitions for medical NLP tasks."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


class NERPredictionType:
    """Container for NER prediction results.

    This class represents the output of a named entity recognition prediction,
    including the predicted labels, their scores, and the corresponding tokens.
    """

    def __init__(
        self,
        entities: List[Dict[str, Union[str, float, Tuple[int, int]]]],
        tokens: List[str],
        text: Optional[str] = None,
    ) -> None:
        """Initialize the NER prediction.

        Args:
            entities: List of entity dictionaries, each containing:
                - 'entity': The entity type (e.g., 'DISEASE', 'TREATMENT')
                - 'score': Confidence score between 0 and 1
                - 'start': Start position in the token list
                - 'end': End position in the token list (exclusive)
            tokens: List of tokens that were processed
            text: Optional original text (if available)
        """
        self.entities = entities
        self.tokens = tokens
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        """Convert the prediction to a dictionary.

        Returns:
            Dictionary representation of the prediction.
        """
        return {
            "entities": self.entities,
            "tokens": self.tokens,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NERPredictionType":
        """Create a prediction from a dictionary.

        Args:
            data: Dictionary containing prediction data.

        Returns:
            NERPredictionType instance.
        """
        return cls(
            entities=data["entities"],
            tokens=data["tokens"],
            text=data.get("text"),
        )


class EntityType(Enum):
    """Enumeration of entity types for medical NER."""

    DISEASE = auto()
    TREATMENT = auto()
    MEDICATION = auto()
    DOSAGE = auto()
    ROUTE = auto()
    FREQUENCY = auto()
    DURATION = auto()
    BODY_PART = auto()
    SYMPTOM = auto()
    PROCEDURE = auto()
    TEST = auto()
    TEST_RESULT = auto()
    PERSON = auto()
    ORGANIZATION = auto()
    DATE = auto()
    TIME = auto()
    AGE = auto()
    GENDER = auto()
    RACE = auto()
    FAMILY = auto()
    OTHER = auto()


@dataclass
class EntitySpan:
    """Represents a span of text identified as a named entity."""

    start: int
    end: int
    label: str
    text: str
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entity span to a dictionary.

        Returns:
            Dictionary representation of the entity span.
        """
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "text": self.text,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntitySpan":
        """Create an entity span from a dictionary.

        Args:
            data: Dictionary containing entity span data.

        Returns:
            EntitySpan instance.
        """
        return cls(
            start=data["start"],
            end=data["end"],
            label=data["label"],
            text=data["text"],
            score=data.get("score", 1.0),
        )
