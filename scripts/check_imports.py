import sys
import os


def check_import(module_name):
    print(f"\nChecking import for: {module_name}")
    try:
        module = __import__(module_name)
        print(f"Successfully imported {module_name}")
        print(
            f"Module path: {os.path.dirname(module.__file__) if hasattr(module, '__file__') else 'built-in'}"
        )

        # Try to import a submodule
        try:
            submodule_name = f"{module_name}.tokenization_utils_base"
            __import__(submodule_name)
            print(f"Successfully imported submodule: {submodule_name}")
        except ImportError as e:
            print(f"Failed to import submodule {submodule_name}: {e}")

    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")


# Check for any transformers modules in the current directory
print("Checking for local transformers modules that might shadow the package:")
for root, dirs, files in os.walk("."):
    if "transformers" in dirs:
        print(f"Found local transformers directory at: {os.path.join(root, 'transformers')}")
    if "transformers.py" in files:
        print(f"Found local transformers.py at: {os.path.join(root, 'transformers.py')}")

# Check the actual imports
check_import("transformers")

# Try to import the specific class directly
print("\nTrying to import PreTrainedTokenizerBase directly:")
try:
    print("Successfully imported PreTrainedTokenizerBase")
except Exception as e:
    print(f"Failed to import PreTrainedTokenizerBase: {e}")
    print("\nPython path:")
    for p in sys.path:
        print(f"- {p}")
