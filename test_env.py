import sys
import os
import importlib

print("Python version:", sys.version)
print("\nPython path:")
for path in sys.path:
    print(f"  {path}")

print("\nTrying to import transformers...")
try:
    # Try to import transformers
    import transformers
    print(f"✅ transformers imported from: {transformers.__file__}")
    print(f"transformers version: {transformers.__version__}")
    
    # Try to import the specific module
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    print("✅ PreTrainedTokenizerBase imported successfully")
    
    # Check if transformers is a package
    print(f"\nIs transformers a package? {hasattr(transformers, '__path__')}")
    if hasattr(transformers, '__path__'):
        print(f"transformers path: {transformers.__path__}")
    
    # Try to import the module directly
    tokenization_utils_base = importlib.import_module('transformers.tokenization_utils_base')
    print(f"✅ Direct import of tokenization_utils_base: {tokenization_utils_base.__file__}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Check if transformers is a file
    transformers_path = None
    for path in sys.path:
        potential_path = os.path.join(path, "transformers.py")
        if os.path.exists(potential_path):
            transformers_path = potential_path
            break
    
    if transformers_path:
        print(f"\n⚠️ Found transformers.py at: {transformers_path}")
        print("This file might be shadowing the transformers package.")
    else:
        print("\n⚠️ No transformers.py found in Python path")
