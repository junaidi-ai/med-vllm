"""
Tests for MedicalModelConfig.copy update dict vs kwargs precedence and errors.
"""

import pytest

from medvllm.medical.config.models.medical_config import MedicalModelConfig


class TestMedicalConfigCopyPrecedence:
    def make_cfg(self, **overrides):
        base = {
            "model": "dummy",
            "model_type": "medical_llm",
            "medical_entity_types": ["disease"],
        }
        base.update(overrides)
        return MedicalModelConfig.from_dict(base)

    def test_copy_with_update_dict(self):
        cfg = self.make_cfg(batch_size=16)
        new_cfg = cfg.copy(update={"batch_size": 64})
        assert new_cfg.batch_size == 64
        assert cfg.batch_size == 16  # original unchanged

    def test_copy_with_kwargs(self):
        cfg = self.make_cfg(batch_size=16)
        new_cfg = cfg.copy(batch_size=32)
        assert new_cfg.batch_size == 32
        assert cfg.batch_size == 16

    def test_copy_kwargs_take_precedence_over_update_dict(self):
        cfg = self.make_cfg(batch_size=16)
        new_cfg = cfg.copy(update={"batch_size": 64}, batch_size=32)
        assert new_cfg.batch_size == 32

    def test_copy_invalid_update_type_raises_type_error(self):
        cfg = self.make_cfg(batch_size=16)
        with pytest.raises(TypeError, match="update must be a dict"):
            cfg.copy(update=[("batch_size", 64)])  # type: ignore[arg-type]

    def test_copy_unknown_attribute_raises(self):
        cfg = self.make_cfg()
        # Depending on whether original_copy exists, this can be AttributeError or TypeError
        with pytest.raises((AttributeError, TypeError)):
            cfg.copy(update={"non_existent_field": 123})
