"""Minimal test to reproduce the import issue."""

import sys
import os

# Print environment info
print("Python executable:", sys.executable)
print("Working directory:", os.getcwd())
print("\nPython path:")
for p in sys.path:
    print(f"  {p}")

# Try to import the test file directly
try:
    print("\nTrying to import test_type_utils.py directly...")
    from tests.unit.config.test_utils import test_type_utils
    print("Successfully imported test_type_utils")
except ImportError as e:
    print(f"Error importing test_type_utils: {e}")
    # Print the traceback for more details
    import traceback
    traceback.print_exc()

# Try to import the problematic module directly
try:
    print("\nTrying to import from medvllm.engine.llm_engine...")
    from medvllm.engine import llm_engine
    print("Successfully imported llm_engine")
except ImportError as e:
    print(f"Error importing llm_engine: {e}")
    import traceback
    traceback.print_exc()

# Try to import transformers directly
try:
    print("\nTrying to import transformers directly...")
    import transformers
    print(f"Successfully imported transformers from: {transformers.__file__}")
    
    print("\nTrying to import tokenization_utils_base...")
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    print("Successfully imported PreTrainedTokenizerBase")
except ImportError as e:
    print(f"Error importing transformers: {e}")
    import traceback
    traceback.print_exc()
