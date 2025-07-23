import sys
import torch
from transformers import AutoModelForSequenceClassification

def test_imports():
    print("\n=== Testing Imports ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {AutoModelForSequenceClassification.__module__}")
    assert True
