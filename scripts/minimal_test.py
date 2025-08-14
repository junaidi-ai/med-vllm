# minimal_test.py
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Test imports
print("Testing imports...")
try:
    import transformers

    print("✅ transformers imported successfully")
    print(f"transformers path: {transformers.__file__}")

    print("✅ PreTrainedTokenizerBase imported successfully")

    # Try importing medvllm
    print("✅ medvllm imported successfully")

    print("✅ LLMEngine imported successfully")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
