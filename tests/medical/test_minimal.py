import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_imports():
    try:
        import transformers
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        assert True
    except Exception as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
        assert False, "Failed to import transformers"
