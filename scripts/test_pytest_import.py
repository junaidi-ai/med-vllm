#!/usr/bin/env python3
"""Test script to diagnose pytest import issues."""

import sys
import os


def test_imports():
    print("\n=== Test function running ===")
    print("Python version:", sys.version)
    print("Python executable:", sys.executable)
    print("Working directory:", os.getcwd())
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")

    print("\nTrying to import transformers...")
    try:
        import transformers

        print(f"Successfully imported transformers from: {transformers.__file__}")
        print(f"Transformers version: {transformers.__version__}")

        print("\nTrying to import tokenization_utils_base...")
        try:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase

            print("Successfully imported PreTrainedTokenizerBase")
            assert True
        except ImportError as e:
            print(f"Error importing PreTrainedTokenizerBase: {e}")
            print("\nContents of transformers directory:")
            transformers_dir = os.path.dirname(transformers.__file__)
            try:
                files = os.listdir(transformers_dir)
                for f in files:
                    if f.startswith("tokenization") or f == "utils":
                        print(f"  {f}")
            except Exception as e:
                print(f"Error listing transformers directory: {e}")
            assert False, f"Failed to import PreTrainedTokenizerBase: {e}"

    except ImportError as e:
        print(f"Error importing transformers: {e}")
        print("\nInstalled packages:")
        try:
            import pkg_resources

            for dist in pkg_resources.working_set:
                if "transformers" in dist.project_name.lower():
                    print(f"  {dist.project_name} ({dist.version}) at {dist.location}")
        except Exception as e:
            print(f"Error getting installed packages: {e}")
        assert False, f"Failed to import transformers: {e}"
