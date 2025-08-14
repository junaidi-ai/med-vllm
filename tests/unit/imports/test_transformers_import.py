#!/usr/bin/env python3
"""
Test script to verify transformers imports work correctly.
"""

import sys
import os


def print_sys_path():
    print("\nPython path:")
    for path in sys.path:
        print(f"- {path}")


def test_imports():
    print("\nTesting imports...")

    # Test importing transformers directly
    try:
        import transformers

        print(f"Successfully imported transformers from: {transformers.__file__}")
    except ImportError as e:
        print(f"Failed to import transformers: {e}")
        return False

    # Test importing the specific module that's failing
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        print("Successfully imported PreTrainedTokenizerBase")
        return True
    except ImportError as e:
        print(f"Failed to import PreTrainedTokenizerBase: {e}")
        return False


if __name__ == "__main__":
    print("Transformers Import Test")
    print("=======================")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    print_sys_path()

    if test_imports():
        print("\n✅ All imports successful!")
    else:
        print("\n❌ Some imports failed.")
