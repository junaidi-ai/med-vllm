import sys
import torch
import transformers
from transformers import AutoModelForSequenceClassification

def test_imports():
    print("\n=== Testing Imports ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Transformers path: {transformers.__file__}")
    
    # Test if we can import the specific module
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        print("✅ Successfully imported PreTrainedTokenizerBase")
        assert True
    except Exception as e:
        print(f"❌ Error importing PreTrainedTokenizerBase: {e}")
        raise
