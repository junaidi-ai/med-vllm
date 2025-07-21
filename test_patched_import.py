import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Apply our patch before any other imports
import warnings
warnings.filterwarnings("ignore", message=".*is already registered.*")

# Import the patch
try:
    import registry_fix
    print("✅ Successfully applied registry patch")
except ImportError as e:
    print(f"❌ Failed to apply registry patch: {e}")
    sys.exit(1)

# Now try to import the problematic module
print("\nTrying to import LLMEngine...")
try:
    from medvllm.engine.llm_engine import LLMEngine
    print("✅ Successfully imported LLMEngine")
except ImportError as e:
    print(f"❌ Failed to import LLMEngine: {e}")
    sys.exit(1)
