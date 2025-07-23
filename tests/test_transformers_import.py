import sys
import os

def test_import_transformers():
    try:
        import transformers
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        assert True
    except Exception as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
        assert False, "Failed to import transformers"
