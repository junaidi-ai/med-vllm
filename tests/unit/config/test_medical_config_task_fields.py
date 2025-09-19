"""
Unit tests for task_type and classification_labels in MedicalModelConfig.

Covers:
- default task_type is 'ner'
- validation: task_type must be one of {classification, ner, generation}
- validation: when task_type == 'classification', classification_labels must be non-empty list of non-empty strings
- validation: labels cannot contain None or empty/whitespace-only strings
- normalization: task_type lowercased; labels stripped and deduplicated preserving order
- roundtrip: to_dict/from_dict preserves task fields
"""

import pytest

from medvllm.medical.config.models.medical_config import MedicalModelConfig


@pytest.mark.unit
class TestMedicalConfigTaskFields:
    def test_default_task_type_is_ner(self, tmp_path):
        cfg = MedicalModelConfig(model=str(tmp_path / "m"))
        assert cfg.task_type == "ner"
        assert cfg.classification_labels == []

    @pytest.mark.parametrize(
        "task_type,valid",
        [
            ("classification", False),  # requires labels
            ("ner", True),
            ("generation", True),
            ("CLASSIFICATION", False),  # normalized but still requires labels
            ("invalid", False),
            (123, False),
        ],
    )
    def test_task_type_validation(self, tmp_path, task_type, valid):
        base = {"model": str(tmp_path / "m"), "task_type": task_type}
        if valid:
            cfg = MedicalModelConfig.from_dict(base)
            assert cfg.task_type in {"classification", "ner", "generation"}
        else:
            with pytest.raises(ValueError):
                MedicalModelConfig.from_dict(base)

    def test_classification_requires_non_empty_labels(self, tmp_path):
        base = {"model": str(tmp_path / "m"), "task_type": "classification"}
        with pytest.raises(ValueError):
            MedicalModelConfig.from_dict(base)

        cfg = MedicalModelConfig.from_dict(
            {
                **base,
                "classification_labels": ["diagnosis", "treatment"],
            }
        )
        assert cfg.classification_labels == ["diagnosis", "treatment"]

    @pytest.mark.parametrize("labels", [None, [""], ["  "], ["x", None], ["a", "", "b"]])
    def test_labels_invalid_values(self, tmp_path, labels):
        base = {
            "model": str(tmp_path / "m"),
            "task_type": "classification",
            "classification_labels": labels,
        }
        with pytest.raises(ValueError):
            MedicalModelConfig.from_dict(base)

    def test_labels_dedup_and_strip(self, tmp_path):
        cfg = MedicalModelConfig.from_dict(
            {
                "model": str(tmp_path / "m"),
                "task_type": "classification",
                "classification_labels": [" Diagnosis ", "treatment", "diagnosis", " follow-up "],
            }
        )
        # stripped and deduplicated preserving first occurrence order
        assert cfg.classification_labels == ["Diagnosis", "treatment", "follow-up"]

    def test_non_classification_can_have_empty_labels(self, tmp_path):
        # ner
        cfg1 = MedicalModelConfig.from_dict({"model": str(tmp_path / "m1"), "task_type": "ner"})
        assert cfg1.classification_labels == []
        # generation
        cfg2 = MedicalModelConfig.from_dict(
            {"model": str(tmp_path / "m2"), "task_type": "generation"}
        )
        assert cfg2.classification_labels == []

    def test_roundtrip_preserves_task_fields(self, tmp_path):
        cfg = MedicalModelConfig.from_dict(
            {
                "model": str(tmp_path / "m"),
                "task_type": "classification",
                "classification_labels": ["diagnosis", "treatment"],
            }
        )
        d = cfg.to_dict()
        # version key is injected for legacy BC; ignore it when reconstructing
        d.pop("version", None)
        cfg2 = MedicalModelConfig.from_dict(d)
        assert cfg2.task_type == "classification"
        assert cfg2.classification_labels == ["diagnosis", "treatment"]
