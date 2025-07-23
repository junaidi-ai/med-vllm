def test_import_working():
    """Test that verifies imports work correctly"""
    import torch
    from transformers import AutoModelForSequenceClassification
    
    print("\n=== Test Imports ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {AutoModelForSequenceClassification.__module__}")
    
    # Just verify the imports worked
    assert True
