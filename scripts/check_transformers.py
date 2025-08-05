import sys
import os

# Print Python path
print("Python path:")
for p in sys.path:
    print(f"- {p}")

# Try to import tokenization_utils_base
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    print("Successfully imported PreTrainedTokenizerBase")
except ImportError as e:
    print(f"Error importing PreTrainedTokenizerBase: {e}")
    print("\nContents of transformers directory:")
    transformers_path = os.path.dirname(os.path.abspath(__file__)) + "/venv/lib/python3.12/site-packages/transformers"
    for root, dirs, files in os.walk(transformers_path):
        level = root.replace(transformers_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
