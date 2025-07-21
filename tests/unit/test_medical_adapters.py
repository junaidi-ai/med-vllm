"""
Comprehensive tests for medical model adapters.
This file consolidates all adapter tests using a clean, maintainable pattern.
"""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import torch
import torch.nn as nn

# ============================================================================
# Base Test Classes and Utilities
# ============================================================================


class SimpleModel(nn.Module):
    """A simple model for testing adapters."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 512,
            "vocab_size": 1000,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize embedding layer
        self.embeddings = nn.Embedding(
            self.config["vocab_size"], self.config["hidden_size"]
        )

        # Initialize a simple transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config["hidden_size"],
            nhead=self.config["num_attention_heads"],
            dim_feedforward=self.config["intermediate_size"],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config["num_hidden_layers"]
        )

        # Pooler layer
        self.pooler = nn.Linear(self.config["hidden_size"], self.config["hidden_size"])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights like BERT."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Get input shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = (1, 5)  # Default shape

        batch_size = input_shape[0]
        seq_length = input_shape[1] if len(input_shape) > 1 else 5

        # Get embeddings
        if input_ids is not None:
            embedding_output = self.embeddings(input_ids)
        else:
            # Create dummy input if not provided
            input_ids = torch.zeros(
                (batch_size, seq_length), dtype=torch.long, device=self.device
            )
            embedding_output = self.embeddings(input_ids)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to float and expand dimensions for broadcasting
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0

        # Pass through encoder
        encoder_outputs = self.encoder(
            embedding_output,
            src_key_padding_mask=(
                attention_mask.squeeze(1).squeeze(1)
                if attention_mask is not None
                else None
            ),
        )

        # Get pooled output (CLS token)
        pooled_output = self.pooler(encoder_outputs[:, 0])

        return {"last_hidden_state": encoder_outputs, "pooler_output": pooled_output}


class SimpleTokenizer:
    """A simple tokenizer for testing."""

    def __init__(self):
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.unk_token_id = 100
        self.vocab_size = 1000
        self.special_tokens_map = {
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "unk_token": "[UNK]",
        }

    def __call__(self, text, *args, **kwargs):
        # Simple tokenization - split on spaces and map to IDs
        words = text.split()
        input_ids = [self.cls_token_id] + [
            hash(word) % (self.vocab_size - 5) + 5 for word in words
        ]
        attention_mask = [1] * len(input_ids)

        # Pad to max_length if specified
        if "max_length" in kwargs and len(input_ids) < kwargs["max_length"]:
            pad_length = kwargs["max_length"] - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }

    def save_pretrained(self, save_directory):
        # Mock save method
        pass


# ============================================================================
# Test Cases
# ============================================================================


class TestBaseAdapter(unittest.TestCase):
    """Tests for the base adapter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.tokenizer = SimpleTokenizer()

        # Create a simple adapter class for testing
        class TestAdapter:
            def __init__(self, model, config=None):
                self.model = model
                self.config = config or {}
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self.model = self.model.to(self.device)

            def to(self, device):
                if isinstance(device, str):
                    device = torch.device(device)
                self.device = device
                self.model = self.model.to(device)
                return self

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        self.adapter_class = TestAdapter
        self.adapter = TestAdapter(self.model)

    def test_adapter_initialization(self):
        """Test that the adapter can be initialized with a model."""
        self.assertIsNotNone(self.adapter.model)
        self.assertEqual(
            self.adapter.device.type, "cuda" if torch.cuda.is_available() else "cpu"
        )

    def test_adapter_forward_pass(self):
        """Test that the adapter can perform a forward pass."""
        # Create input tensor
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        # Test forward pass
        with torch.no_grad():
            outputs = self.adapter.forward(input_ids=input_ids)

        # Verify output shapes
        self.assertIn("last_hidden_state", outputs)
        self.assertIn("pooler_output", outputs)
        self.assertEqual(outputs["last_hidden_state"].shape, (1, 5, 128))
        self.assertEqual(outputs["pooler_output"].shape, (1, 128))

        # Verify output types
        self.assertIsInstance(outputs["last_hidden_state"], torch.Tensor)
        self.assertIsInstance(outputs["pooler_output"], torch.Tensor)

        # Verify device
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(outputs["last_hidden_state"].device, expected_device)
        self.assertEqual(outputs["pooler_output"].device, expected_device)


class TestBioBERTAdapter(unittest.TestCase):
    """Tests for the BioBERT adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.tokenizer = SimpleTokenizer()

        # Create a mock BioBERT adapter
        class MockBioBERTAdapter:
            def __init__(self, model, config=None):
                self.model = model
                self.config = config or {}
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self.model = self.model.to(self.device)

                # BioBERT specific attributes
                self.biomedical_terms = [
                    "diabetes",
                    "hypertension",
                    "myocardial",
                    "infarction",
                    "tachycardia",
                    "bradycardia",
                    "pneumonia",
                    "asthma",
                ]

            def to(self, device):
                if isinstance(device, str):
                    device = torch.device(device)
                self.device = device
                self.model = self.model.to(device)
                return self

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

            def preprocess_biomedical_text(self, text):
                """Mock preprocessing for biomedical text."""
                # Simple preprocessing - just add a prefix
                return f"[BIOMEDICAL] {text}"

        self.adapter_class = MockBioBERTAdapter
        self.adapter = MockBioBERTAdapter(self.model)

    def test_biomedical_text_processing(self):
        """Test biomedical text preprocessing."""
        text = "Patient has diabetes and hypertension."
        processed = self.adapter.preprocess_biomedical_text(text)
        self.assertTrue(processed.startswith("[BIOMEDICAL]"))
        self.assertIn("diabetes", processed)
        self.assertIn("hypertension", processed)

    def test_forward_pass_with_biomedical_terms(self):
        """Test forward pass with biomedical terms in input."""
        # Create input with biomedical terms
        input_text = "Patient has diabetes and hypertension."
        tokenized = self.tokenizer(input_text, max_length=10)

        # Test forward pass
        with torch.no_grad():
            outputs = self.adapter.forward(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )

        # Verify outputs
        self.assertIn("last_hidden_state", outputs)
        self.assertIn("pooler_output", outputs)
        self.assertEqual(outputs["last_hidden_state"].shape[0], 1)  # batch size 1
        self.assertEqual(outputs["pooler_output"].shape[0], 1)  # batch size 1


class TestClinicalBERTAdapter(unittest.TestCase):
    """Tests for the ClinicalBERT adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.tokenizer = SimpleTokenizer()

        # Create a mock ClinicalBERT adapter
        class MockClinicalBERTAdapter:
            def __init__(self, model, config=None):
                self.model = model
                self.config = config or {}
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self.model = self.model.to(self.device)

                # ClinicalBERT specific attributes
                self.clinical_terms = [
                    "admission",
                    "discharge",
                    "symptoms",
                    "diagnosis",
                    "treatment",
                    "medication",
                    "allergies",
                    "prognosis",
                ]

            def to(self, device):
                if isinstance(device, str):
                    device = torch.device(device)
                self.device = device
                self.model = self.model.to(device)
                return self

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

            def preprocess_clinical_text(self, text, note_type=None):
                """Mock preprocessing for clinical text."""
                prefix = f"[{note_type.upper()}] " if note_type else ""
                return f"{prefix}{text}"

        self.adapter_class = MockClinicalBERTAdapter
        self.adapter = MockClinicalBERTAdapter(self.model)

    def test_clinical_text_processing(self):
        """Test clinical text preprocessing."""
        text = "Patient admitted with chest pain."

        # Test with note type
        processed = self.adapter.preprocess_clinical_text(text, "admission")
        self.assertTrue(processed.startswith("[ADMISSION]"))

        # Test without note type
        processed = self.adapter.preprocess_clinical_text(text)
        self.assertIn("chest pain", processed)

    def test_forward_pass_with_clinical_terms(self):
        """Test forward pass with clinical terms in input."""
        # Create input with clinical terms
        input_text = "Patient admitted with chest pain and SOB."
        tokenized = self.tokenizer(input_text, max_length=15)

        # Test forward pass
        with torch.no_grad():
            outputs = self.adapter.forward(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )

        # Verify outputs
        self.assertIn("last_hidden_state", outputs)
        self.assertIn("pooler_output", outputs)
        self.assertEqual(outputs["last_hidden_state"].shape[0], 1)  # batch size 1
        self.assertEqual(outputs["pooler_output"].shape[0], 1)  # batch size 1


# ============================================================================
# Test Runner
# ============================================================================


def run_tests():
    """Run all tests."""
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBaseAdapter))
    test_suite.addTest(unittest.makeSuite(TestBioBERTAdapter))
    test_suite.addTest(unittest.makeSuite(TestClinicalBERTAdapter))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(test_suite)


if __name__ == "__main__":
    # Run tests directly when executed as a script
    unittest.main()
