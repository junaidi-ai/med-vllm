"""Script to check imports and basic functionality."""

import sys

import torch
from transformers import AutoModel, AutoTokenizer


def main():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("Transformers version:")

    # Try to import AutoModel and AutoTokenizer
    try:
        from transformers import AutoModel, AutoTokenizer

        print("Successfully imported AutoModel and AutoTokenizer")

        # Test model loading
        print("\nTesting BioBERT model loading...")
        model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        print("Successfully loaded BioBERT model and tokenizer")

        # Test forward pass
        inputs = tokenizer("This is a test sentence.", return_tensors="pt")
        outputs = model(**inputs)
        print("Forward pass successful!")
        print(f"Output shape: {outputs.last_hidden_state.shape}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
