"""Test script to check imports and basic functionality."""

import os
import sys

print("Python version:", sys.version)
print("Python path:", sys.path)
print("Current working directory:", os.getcwd())

# Try importing transformers
try:
    import transformers

    print("Successfully imported transformers")
    print("Transformers version:", transformers.__version__)
    print("Transformers path:", transformers.__file__)

    # Try importing the tokenizer base class
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    print("Successfully imported PreTrainedTokenizerBase")

    # Try importing the model registry
    try:
        from medvllm.engine.model_runner.registry import ModelRegistry, ModelType

        print("Successfully imported ModelRegistry and ModelType")
    except ImportError as e:
        print(f"Error importing ModelRegistry: {e}")

except ImportError as e:
    print(f"Error importing transformers: {e}")

    # Print more debug info
    print("\nTransformers package info:")
    try:
        import pkg_resources

        dist = pkg_resources.get_distribution("transformers")
        print(f"Transformers package location: {dist.location}")
        print(f"Transformers package files: {dist.get_metadata('top_level.txt')}")
    except Exception as e:
        print(f"Could not get package info: {e}")

print("\nEnvironment variables:")
for key in ["PYTHONPATH", "PATH"]:
    print(f"{key}: {os.environ.get(key, 'Not set')}")
