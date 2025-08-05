"""Minimal test file to isolate the import issue."""

def test_import_transformers():
    """Test importing transformers directly."""
    import sys
    import os
    
    print("\n=== Test function running ===")
    print("Python executable:", sys.executable)
    print("Working directory:", os.getcwd())
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")
    
    print("\nTrying to import transformers...")
    import transformers
    print(f"Successfully imported transformers from: {transformers.__file__}")
    print(f"Transformers version: {transformers.__version__}")
    
    print("\nTrying to import tokenization_utils_base...")
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    print("Successfully imported PreTrainedTokenizerBase")
    
    assert True  # If we get here, the test passes
