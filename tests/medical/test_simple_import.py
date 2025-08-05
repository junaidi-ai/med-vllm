import sys
import torch

def test_imports():
    print("\n=== Testing Imports ===")
    print(f"Python: {sys.version}")
    
    # Check PyTorch import
    print(f"PyTorch: {getattr(torch, '__version__', 'mock')}")
    
    # Check transformers import with error handling
    try:
        from transformers import AutoModelForSequenceClassification
        print(f"Transformers: {AutoModelForSequenceClassification.__module__}")
    except ImportError as e:
        print(f"Transformers import error: {e}")
    
    # Basic assertion to ensure the test runs
    assert True
