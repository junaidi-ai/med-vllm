"""Integration tests for the BioBERT adapter."""

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from medvllm.models.adapter import BioBERTAdapter, create_medical_adapter
from medvllm.models.medical_models import BioBERTLoader


@pytest.fixture
def biobert_model_and_tokenizer():
    """Fixture that loads a small BioBERT model and its tokenizer for testing."""
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load a small model for testing
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.float32,
    )

    return model, tokenizer


def test_biobert_adapter_initialization(biobert_model_and_tokenizer):
    """Test that the BioBERT adapter initializes correctly."""
    model, _ = biobert_model_and_tokenizer
    adapter = BioBERTAdapter(model=model, config={"num_labels": 2})

    assert adapter is not None
    assert adapter.model == model
    assert adapter.model_type == "biobert"


def test_biobert_adapter_forward(biobert_model_and_tokenizer):
    """Test the forward pass of the BioBERT adapter."""
    model, tokenizer = biobert_model_and_tokenizer
    adapter = BioBERTAdapter(model=model, config={"num_labels": 2})

    # Test input
    text = "Patient presents with fever and cough."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass
    with torch.no_grad():
        outputs = adapter(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

    # Check output shape
    assert outputs.shape == (1, 2)  # batch_size=1, num_labels=2


def test_biobert_adapter_kv_cache(biobert_model_and_tokenizer):
    """Test KV caching in the BioBERT adapter."""
    model, tokenizer = biobert_model_and_tokenizer
    adapter = BioBERTAdapter(model=model, config={"num_labels": 2})

    # Set up for inference with KV cache
    adapter.setup_for_inference(use_cuda_graphs=False)

    # First pass - should populate KV cache
    text1 = "Patient has a history of"
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs1 = adapter(
            input_ids=inputs1["input_ids"], attention_mask=inputs1["attention_mask"]
        )

    # Second pass - should use KV cache
    text2 = " diabetes and hypertension."
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs2 = adapter(
            input_ids=inputs2["input_ids"], attention_mask=inputs2["attention_mask"]
        )

    # Check that outputs are valid
    assert outputs1.shape == (1, 2)
    assert outputs2.shape == (1, 2)


def test_biobert_loader_with_adapter():
    """Test that the BioBERT loader works with the adapter."""
    # Load model and tokenizer using the loader
    loader = BioBERTLoader()
    model, tokenizer = loader.load_model(
        model_class=AutoModelForSequenceClassification,
        config={"num_labels": 2},
        device="cpu",
    )

    # Create adapter
    adapter = create_medical_adapter(
        model=model, model_type="biobert", config={"num_labels": 2}
    )

    # Test forward pass
    text = "The patient was prescribed aspirin."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = adapter(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

    assert outputs.shape == (1, 2)  # batch_size=1, num_labels=2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
