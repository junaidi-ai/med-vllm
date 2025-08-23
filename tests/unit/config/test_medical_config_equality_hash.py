"""
Tests for patched equality (__eq__) and hashing (__hash__) on MedicalModelConfig.

These tests assert:
- Equality is based on the stable dict representation (patched __eq__)
- Hash values are consistent for equal configs and usable in sets/dicts (patched __hash__)
- Hashing is robust even with self-referential attributes
"""

import pytest

from medvllm.medical.config.models.medical_config import MedicalModelConfig


pytestmark = pytest.mark.unit


def make_cfg(**overrides):
    base = {
        "model": "dummy",
        "model_type": "medical_llm",
        "medical_entity_types": ["disease"],
    }
    base.update(overrides)
    return MedicalModelConfig.from_dict(base)


class TestMedicalModelConfigEqualityHash:
    def test_equality_same_content(self):
        cfg1 = make_cfg(batch_size=16)
        cfg2 = make_cfg(batch_size=16)
        assert cfg1 == cfg2
        assert cfg1.to_dict() == cfg2.to_dict()

    def test_inequality_different_content(self):
        cfg1 = make_cfg(batch_size=16)
        cfg3 = make_cfg(batch_size=32)
        assert cfg1 != cfg3
        assert cfg1.to_dict() != cfg3.to_dict()

    def test_hash_equality_for_equal_instances(self):
        cfg1 = make_cfg(batch_size=16)
        cfg2 = make_cfg(batch_size=16)
        # Equal objects should have equal hashes
        assert hash(cfg1) == hash(cfg2)
        # Usable as dict keys / set elements
        s = {cfg1, cfg2}
        assert len(s) == 1

    def test_hash_changes_when_content_changes(self):
        cfg1 = make_cfg(batch_size=16)
        cfg2 = make_cfg(batch_size=32)
        # Very unlikely to collide; primary assertion is inequality and set behavior
        s = {cfg1, cfg2}
        assert len(s) == 2

    def test_stable_hash_across_calls(self):
        cfg = make_cfg(batch_size=16)
        h1 = hash(cfg)
        h2 = hash(cfg)
        assert h1 == h2

    def test_hash_with_self_reference_attribute(self):
        cfg = make_cfg(batch_size=16)
        # Add a self-referential attribute; patched hashing should tolerate this safely
        cfg.self_ref = cfg
        # Should not raise and should return an int
        h = hash(cfg)
        assert isinstance(h, int)

    def test_not_equal_to_non_config(self):
        cfg = make_cfg()
        assert (cfg == object()) is False
