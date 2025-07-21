import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath("."))

print("Python path:")
print("\n".join(sys.path))
print("\nEnvironment variables:")
for k, v in os.environ.items():
    if "PYTHON" in k or "PATH" in k:
        print(f"{k}: {v}")

print("\nTrying to import from llm_engine...")
try:
    from medvllm.engine.llm_engine import LLMEngine

    print("✅ Successfully imported LLMEngine")
except ImportError as e:
    print(f"❌ Failed to import LLMEngine: {e}")
    print("\nTrying to import transformers directly...")
    try:
        import transformers

        print(f"✅ Successfully imported transformers from: {transformers.__file__}")
        print(f"Transformers version: {transformers.__version__}")
        print("\nTrying to import tokenization_utils_base...")
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        print("✅ Successfully imported PreTrainedTokenizerBase")
    except ImportError as e:
        print(f"❌ Failed to import transformers: {e}")
