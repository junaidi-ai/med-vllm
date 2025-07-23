# test_import.py
import sys
print("Python path:", sys.path)
print("---")

try:
    import transformers
    print("transformers imported successfully")
    print("transformers path:", transformers.__file__)
    
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    print("Successfully imported PreTrainedTokenizerBase")
    
except Exception as e:
    print(f"Error: {e}")
    print("---")
    import traceback
    traceback.print_exc()
