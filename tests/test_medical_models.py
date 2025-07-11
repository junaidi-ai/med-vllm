"""Tests for medical model integration."""

import os
import sys
import unittest
from typing import Any, Dict, Optional, Type, Union
from unittest.mock import MagicMock, Mock, patch

# Mock the torch and transformers modules before importing anything that might use them
sys.modules["torch"] = MagicMock()
sys.modules["torch.optim"] = MagicMock()
sys.modules["torch.optim.optimizer"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.parameter"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.cuda.amp"] = MagicMock()
sys.modules["torch.cuda.amp.grad_scaler"] = MagicMock()
sys.modules["torch.distributed"] = MagicMock()
sys.modules["torch.distributed.fsdp"] = MagicMock()

# Mock transformers
sys.modules["transformers"] = MagicMock()
sys.modules["transformers.modeling_utils"] = MagicMock()
sys.modules["transformers.tokenization_utils_base"] = MagicMock()
sys.modules["transformers.models.auto"] = MagicMock()
sys.modules["transformers.models.auto.modeling_auto"] = MagicMock()
sys.modules["transformers.models.auto.tokenization_auto"] = MagicMock()
sys.modules["transformers.models.auto.configuration_auto"] = MagicMock()

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from medvllm.engine.model_runner.exceptions import (
    ModelLoadingError,
    ModelRegistrationError,
)

# Now import our modules
from medvllm.engine.model_runner.registry import ModelMetadata, ModelRegistry, ModelType


# Mock the MedicalModelLoader base class and its subclasses
class MockMedicalModelLoader:
    """Base class for medical model loaders."""

    @classmethod
    def load_model(cls, device=None, **kwargs):
        return Mock(), Mock()  # Return (model, tokenizer) tuple


class MockBioBERTLoader(MockMedicalModelLoader):
    MODEL_NAME = "dmis-lab/biobert-v1.1"
    MODEL_TYPE = "biomedical"  # Fixed typo in MODEL_TYPE

    @classmethod
    def load_model(cls, device=None, **kwargs):
        return Mock(), Mock()  # Return (model, tokenizer) tuple


class MockClinicalBERTLoader(MockMedicalModelLoader):
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    MODEL_TYPE = "clinical"

    @classmethod
    def load_model(cls, device=None, **kwargs):
        return Mock(), Mock()  # Return (model, tokenizer) tuple


# Patch the imports
sys.modules["medvllm.models"] = MagicMock()
sys.modules["medvllm.models.medical_models"] = MagicMock()
sys.modules["medvllm.models.medical_models"].MedicalModelLoader = MockMedicalModelLoader
sys.modules["medvllm.models.medical_models"].BioBERTLoader = MockBioBERTLoader
sys.modules["medvllm.models.medical_models"].ClinicalBERTLoader = MockClinicalBERTLoader
sys.modules["medvllm.models"].BioBERTLoader = MockBioBERTLoader
sys.modules["medvllm.models"].ClinicalBERTLoader = MockClinicalBERTLoader


# Mock the ModelMetadata class
class MockModelMetadata:
    def __init__(self, name, model_type, tags=None, **kwargs):
        self.name = name
        self.model_type = model_type
        self.tags = tags or []
        self.parameters = kwargs.get("parameters", {})
        self.loader = kwargs.get("loader")
        self.description = kwargs.get("description", "")
        self.model_class = kwargs.get("model_class")
        self.config_class = kwargs.get("config_class")

    def to_dict(self):
        """Convert metadata to a dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "description": self.description,
            "tags": self.tags,
            "parameters": self.parameters,
        }


class TestMedicalModels(unittest.TestCase):
    """Test cases for medical model integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a fresh registry for each test
        self.registry = ModelRegistry()

        # Create mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()

        # Patch the registry's _register_medical_models method to register our mock models
        self.patchers = [
            patch(
                "medvllm.engine.model_runner.registry.MEDICAL_MODELS_AVAILABLE", True
            ),
            patch(
                "medvllm.engine.model_runner.registry.MedicalModelLoader",
                MockMedicalModelLoader,
            ),
            patch(
                "medvllm.engine.model_runner.registry.ModelMetadata", MockModelMetadata
            ),
            patch(
                "medvllm.engine.model_runner.registry.BioBERTLoader", MockBioBERTLoader
            ),
            patch(
                "medvllm.engine.model_runner.registry.ClinicalBERTLoader",
                MockClinicalBERTLoader,
            ),
        ]

        # Start all patchers
        for patcher in self.patchers:
            patcher.start()

        # Manually register the mock models
        self._mock_register_medical_models()

    def _mock_register_medical_models(self):
        """Mock the registration of medical models."""
        # Clear any existing models
        self.registry._models.clear()
        self.registry._model_cache.clear()

        # Register BioBERT by directly adding to _models to avoid registration logic
        self.registry._models["biobert-base"] = MockModelMetadata(
            name="biobert-base",
            model_type=ModelType.BIOMEDICAL,
            description="BioBERT model for biomedical text",
            tags=["biomedical", "bert"],
            loader=MockBioBERTLoader,
            parameters={"pretrained_model_name_or_path": "dmis-lab/biobert-v1.1"},
        )

        # Register Clinical BERT by directly adding to _models
        self.registry._models["clinical-bert-base"] = MockModelMetadata(
            name="clinical-bert-base",
            model_type=ModelType.CLINICAL,
            description="Clinical BERT model for clinical text",
            tags=["clinical", "bert"],
            loader=MockClinicalBERTLoader,
            parameters={
                "pretrained_model_name_or_path": "emilyalsentzer/Bio_ClinicalBERT"
            },
        )

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop all patchers
        for patcher in self.patchers:
            patcher.stop()

        self.registry.clear()

    def test_register_medical_models(self):
        """Test registering medical models."""
        # Clear and re-register models to test the registration flow
        self.registry._models.clear()
        self.registry._model_cache.clear()

        # Register the models
        self.registry._models["biobert-base"] = MockModelMetadata(
            name="biobert-base",
            model_type=ModelType.BIOMEDICAL,
            description="BioBERT model for biomedical text",
            tags=["biomedical", "bert"],
            loader=MockBioBERTLoader,
            parameters={"pretrained_model_name_or_path": "dmis-lab/biobert-v1.1"},
        )

        self.registry._models["clinical-bert-base"] = MockModelMetadata(
            name="clinical-bert-base",
            model_type=ModelType.CLINICAL,
            description="Clinical BERT model for clinical text",
            tags=["clinical", "bert"],
            loader=MockClinicalBERTLoader,
            parameters={
                "pretrained_model_name_or_path": "emilyalsentzer/Bio_ClinicalBERT"
            },
        )

        # Check that models are registered
        self.assertTrue(
            self.registry.is_registered("biobert-base"),
            "BioBERT model should be registered",
        )
        self.assertTrue(
            self.registry.is_registered("clinical-bert-base"),
            "Clinical BERT model should be registered",
        )

        # Check model metadata
        biobert_meta = self.registry.get_metadata("biobert-base")
        self.assertEqual(
            biobert_meta.model_type,
            ModelType.BIOMEDICAL,
            "BioBERT model type should be BIOMEDICAL",
        )
        self.assertIn(
            "biomedical", biobert_meta.tags, "BioBERT tags should include 'biomedical'"
        )

        clinical_meta = self.registry.get_metadata("clinical-bert-base")
        self.assertEqual(
            clinical_meta.model_type,
            ModelType.CLINICAL,
            "Clinical BERT model type should be CLINICAL",
        )
        self.assertIn(
            "clinical",
            clinical_meta.tags,
            "Clinical BERT tags should include 'clinical'",
        )

    def test_load_biobert_model(self):
        """Test loading BioBERT model."""
        # Mock the loader's load_model method to return a tuple
        with patch.object(MockBioBERTLoader, "load_model") as mock_load:
            mock_load.return_value = (self.mock_model, self.mock_tokenizer)

            # Load the model - registry will unpack the tuple and return just the model
            model = self.registry.load_model("biobert-base", device="cpu")

            # Verify the model was loaded
            self.assertIsNotNone(model, "Model should be loaded successfully")
            self.assertIs(model, self.mock_model, "Should return the mock model")
            mock_load.assert_called_once()

            # Verify the device was passed correctly
            args, kwargs = mock_load.call_args
            self.assertEqual(
                kwargs.get("device"), "cpu", "Device should be passed to loader"
            )

            # Check that model was cached
            self.assertIn(
                "biobert-base", self.registry._model_cache, "Model should be cached"
            )

    def test_load_clinical_bert_model(self):
        """Test loading ClinicalBERT model."""
        # Mock the loader's load_model method to return a tuple
        with patch.object(MockClinicalBERTLoader, "load_model") as mock_load:
            mock_load.return_value = (self.mock_model, self.mock_tokenizer)

            # Load the model - registry will unpack the tuple and return just the model
            model = self.registry.load_model("clinical-bert-base", device="cuda")

            # Verify the model was loaded
            self.assertIsNotNone(model, "Model should be loaded successfully")
            self.assertIs(model, self.mock_model, "Should return the mock model")
            mock_load.assert_called_once()

            # Verify the device was passed correctly
            args, kwargs = mock_load.call_args
            self.assertEqual(
                kwargs.get("device"), "cuda", "Device should be passed to loader"
            )

            # Check that model was cached
            self.assertIn(
                "clinical-bert-base",
                self.registry._model_cache,
                "Model should be cached",
            )

    def test_load_model_with_custom_params(self):
        """Test loading model with custom parameters."""
        # Mock the loader's load_model method to return a tuple
        with patch.object(MockBioBERTLoader, "load_model") as mock_load:
            mock_load.return_value = (self.mock_model, self.mock_tokenizer)

            # Load with custom parameters - registry will unpack the tuple and return just the model
            custom_params = {"num_labels": 3, "output_attentions": True}
            model = self.registry.load_model("biobert-base", **custom_params)

            # Verify the model was loaded
            self.assertIsNotNone(model, "Model should be loaded successfully")
            self.assertIs(model, self.mock_model, "Should return the mock model")

            # Verify custom parameters were passed
            mock_load.assert_called_once()
            args, kwargs = mock_load.call_args
            self.assertEqual(
                kwargs.get("num_labels"), 3, "num_labels should be passed to the loader"
            )
            self.assertTrue(
                kwargs.get("output_attentions"),
                "output_attentions should be passed to the loader",
            )

    def test_load_nonexistent_model(self):
        """Test loading a non-existent model."""
        # Clear the models to ensure the model doesn't exist
        self.registry._models.clear()
        self.registry._model_cache.clear()

        # Patch the _load_directly method to raise ModelLoadingError
        with patch.object(self.registry, "_load_directly") as mock_load_directly:
            mock_load_directly.side_effect = ModelLoadingError(
                "Model 'nonexistent-model' not found in registry or hub",
                model_name="nonexistent-model",
            )

            with self.assertRaises(
                ModelLoadingError,
                msg="Loading non-existent model should raise ModelLoadingError",
            ) as context:
                self.registry.load_model("nonexistent-model")

            # Verify the error message
            self.assertIn(
                "not found",
                str(context.exception).lower(),
                "Error message should indicate model not found",
            )
            self.assertEqual(
                context.exception.model_name,
                "nonexistent-model",
                "Error should include the model name",
            )

    def test_medical_models_not_available(self):
        """Test behavior when medical models are not available."""
        # Stop the patches
        for patcher in self.patchers:
            patcher.stop()

        try:
            # Patch MEDICAL_MODELS_AVAILABLE to False
            with patch(
                "medvllm.engine.model_runner.registry.MEDICAL_MODELS_AVAILABLE", False
            ):
                # Create a mock metadata object and add it to the registry
                self.registry._models["biobert-base"] = MockModelMetadata(
                    name="biobert-base",
                    model_type=ModelType.BIOMEDICAL,
                    tags=["biomedical", "bert"],
                    loader=MockBioBERTLoader,
                )

                # Try to load the model
                with self.assertRaises(ModelLoadingError) as context:
                    self.registry.load_model("biobert-base")

                self.assertIn(
                    "Medical models are not available",
                    str(context.exception),
                    "Should raise an error when medical models are not available",
                )
        finally:
            # Always restore the patches
            for patcher in self.patchers:
                patcher.start()


if __name__ == "__main__":
    unittest.main()
