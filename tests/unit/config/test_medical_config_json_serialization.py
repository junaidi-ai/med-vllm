"""
JSON serialization/deserialization tests for patched MedicalModelConfig.

Covers:
- to_json produces a JSON string from to_dict
- Roundtrip via json.loads + from_dict preserves equality and content
- Enum normalization survives roundtrip
"""

import json
import pytest

from medvllm.medical.config.models.medical_config import MedicalModelConfig
from medvllm.medical.config.types.enums import EntityType


pytestmark = pytest.mark.unit


def test_json_roundtrip_preserves_equality_and_content():
    cfg = MedicalModelConfig.from_dict(
        {
            "model": "dummy",
            "model_type": "medical_llm",
            # Mixed enum representations
            "medical_entity_types": [EntityType.DISEASE, "EntityType.SYMPTOM", "treatment"],
            # Include some non-defaults to ensure they persist
            "batch_size": 7,
            "enable_uncertainty_estimation": True,
            "uncertainty_threshold": 0.42,
            "entity_linking": {"enabled": True, "confidence_threshold": 0.75},
        }
    )

    json_str = cfg.to_json()
    assert isinstance(json_str, str) and json_str.strip().startswith("{")

    loaded_dict = json.loads(json_str)
    cfg2 = MedicalModelConfig.from_dict(loaded_dict)

    # Equality is patched to compare to_dict representations
    assert cfg2 == cfg
    assert cfg2.to_dict() == cfg.to_dict()

    # Ensure enum normalization to lowercase strings
    assert set(cfg2.to_dict()["medical_entity_types"]) == {"disease", "symptom", "treatment"}
