def test_minimal():
    """Minimal working test to verify pytest setup"""
    import torch
    from transformers import AutoModelForSequenceClassification
    
    print("\n=== Running Minimal Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {AutoModelForSequenceClassification.__module__}")
    
    # Simple assertion to verify the test runs
    assert 1 + 1 == 2
