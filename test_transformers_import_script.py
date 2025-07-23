import sys
print("Python path:", sys.path)
print("---")

try:
    import transformers
    print("✅ transformers imported successfully")
    print(f"transformers path: {transformers.__file__}")
    print(f"transformers version: {transformers.__version__}")
    
    # Try to import the specific module
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    print("✅ PreTrainedTokenizerBase imported successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
