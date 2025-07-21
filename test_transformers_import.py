import sys

print("Python path:")
print("\n".join(sys.path))
print("\nTrying to import transformers...")
import transformers

print(f"Successfully imported transformers from: {transformers.__file__}")
print(f"Transformers version: {transformers.__version__}")
print("\nTrying to import PreTrainedTokenizerBase...")
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

print("Successfully imported PreTrainedTokenizerBase")
