"""
Tests for legacy field handling and enum normalization in MedicalModelConfig.
"""

import sys
from typing import List

import pytest

from medvllm.medical.config.models.medical_config import MedicalModelConfig


class TestMedicalConfigLegacyFields:
    """Verify legacy inputs are handled and serialized consistently."""

    def test_from_dict_maps_legacy_fields_and_normalizes_enums(self):
        # Provide legacy fields and mixed enum-like inputs
        cfg_in = {
            "model_name_or_path": "/tmp/model-dir",
            "domain_config": {"obsolete": True},
            "medical_entity_types": [
                "EntityType.DISEASE",
                "Symptom",
                "treatment",
            ],
            "config_version": "1.0.0",
        }

        cfg = MedicalModelConfig.from_dict(cfg_in)

        # Ensure the instance is created and model was mapped
        assert getattr(cfg, "model", None) == "/tmp/model-dir"
        # domain_config should be ignored/removed
        assert not hasattr(cfg, "domain_config") or getattr(cfg, "domain_config") in (None, {})

        # to_dict should normalize values and include defaults
        d = cfg.to_dict()

        # Version behavior: version is present and normalized to 0.1.0 for backward-compat tests
        assert d.get("version") == "0.1.0"

        # Model mapping from legacy key
        assert d.get("model") == "/tmp/model-dir"
        assert "model_name_or_path" not in d

        # Enum-like normalization is lowercase strings
        ents: List[str] = d.get("medical_entity_types", [])
        assert set(ents) >= {"disease", "symptom", "treatment"}

        # Required defaults present
        for key in (
            "document_types",
            "section_headers",
            "batch_size",
            "ner_confidence_threshold",
            "entity_linking",
            "max_entity_span_length",
            "max_retries",
            "request_timeout",
        ):
            assert key in d

    def test_to_dict_sets_model_from_pretrained_if_missing(self):
        # When pretrained_model_name_or_path is present, to_dict should mirror to model if missing
        cfg = MedicalModelConfig.from_dict(
            {
                "pretrained_model_name_or_path": "dmis-lab/biobert-v1.1",
                "medical_entity_types": ["disease"],
            }
        )
        d = cfg.to_dict()
        assert d.get("model") == "dmis-lab/biobert-v1.1"
        assert d.get("version") == "0.1.0"
