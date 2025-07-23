import sys
import os
import importlib

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nTrying to import transformers...")
try:
    # Try to import transformers
    import transformers
    print(f"✅ transformers imported from: {transformers.__file__}")
    
    # Try to import the specific module
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    print("✅ PreTrainedTokenizerBase imported successfully")
    
except Exception as e:
    print(f"❌ Error importing transformers: {e}")
    import traceback
    traceback.print_exc()
    
    # Try to find the transformers package
    print("\nSearching for transformers in site-packages...")
    for path in sys.path:
        if "site-packages" in path:
            transformers_path = os.path.join(path, "transformers")
            if os.path.exists(transformers_path):
                print(f"Found transformers at: {transformers_path}")
                if "tokenization_utils_base.py" in os.listdir(transformers_path):
                    print("  - tokenization_utils_base.py exists")
                else:
                    print("  - tokenization_utils_base.py NOT found")
                if "__init__.py" in os.listdir(transformers_path):
                    print("  - __init__.py exists")
                else:
                    print("  - __init__.py NOT found")
