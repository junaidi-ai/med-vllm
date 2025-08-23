"""Unit tests for `create_medical_adapter` factory using mocked adapters.

These tests verify that the factory returns the specific mocked adapter classes
for supported adapter types and both supported call signatures. They also verify
fallback behavior in `AdapterManager.create_adapter` returns a BioBERT adapter
instance when creation fails.
"""

from unittest.mock import MagicMock

import pytest

from medvllm.models.adapter import create_medical_adapter
from medvllm.models.adapter_manager import AdapterManager
from medvllm.models import adapters as adapters_mod


class TestCreateMedicalAdapterFactory:
    def test_returns_biobert_adapter_when_type_first_signature(self):
        model = MagicMock()
        adapter = create_medical_adapter("biobert", model, {"foo": "bar"})
        assert isinstance(
            adapter, adapters_mod.BioBERTAdapter
        ), "Factory should return BioBERTAdapter mock"
        assert getattr(adapter, "model", None) is model
        assert adapter.config.get("model_type") == "biobert"

    def test_returns_biobert_adapter_when_model_first_signature(self):
        model = MagicMock()
        adapter = create_medical_adapter(model, "biobert", {"foo": "bar"})
        assert isinstance(adapter, adapters_mod.BioBERTAdapter)
        assert getattr(adapter, "model", None) is model
        assert adapter.config.get("model_type") == "biobert"

    def test_returns_clinicalbert_adapter_when_type_first_signature(self):
        model = MagicMock()
        adapter = create_medical_adapter("clinicalbert", model, {"foo": "bar"})
        assert isinstance(adapter, adapters_mod.ClinicalBERTAdapter)
        assert getattr(adapter, "model", None) is model
        assert adapter.config.get("model_type") == "clinicalbert"

    def test_returns_clinicalbert_adapter_when_model_first_signature(self):
        model = MagicMock()
        adapter = create_medical_adapter(model, "clinicalbert", {"foo": "bar"})
        assert isinstance(adapter, adapters_mod.ClinicalBERTAdapter)
        assert getattr(adapter, "model", None) is model
        assert adapter.config.get("model_type") == "clinicalbert"

    @pytest.mark.parametrize(
        "alias, expected_cls",
        [
            ("dmis-lab/biobert", adapters_mod.BioBERTAdapter),
            ("clinical_bert", adapters_mod.ClinicalBERTAdapter),
            ("emilyalsentzer/bio_clinicalbert", adapters_mod.ClinicalBERTAdapter),
        ],
    )
    def test_supported_aliases(self, alias, expected_cls):
        model = MagicMock()
        adapter = create_medical_adapter(alias, model, {"foo": "bar"})
        assert isinstance(adapter, expected_cls)

    def test_unsupported_type_raises(self):
        model = MagicMock()
        with pytest.raises(ValueError, match="Unsupported model type"):
            _ = create_medical_adapter("unknown_adapter", model, {})


class TestAdapterManagerFallbackReturnsMock:
    def test_fallback_returns_biobert_instance(self):
        # No patching: let `create_medical_adapter` raise on unknown type,
        # AdapterManager should fallback and return a BioBERT adapter instance.
        model = MagicMock()
        adapter = AdapterManager.create_adapter(
            model=model,
            model_name_or_path="some-unknown-model",
            adapter_type="unknown",  # force failure on first attempt
            adapter_config=None,  # let manager fill defaults
            hf_config=None,
        )
        assert isinstance(adapter, adapters_mod.BioBERTAdapter)
        assert getattr(adapter, "model", None) is model
        assert adapter.config.get("model_type") == "biobert"
