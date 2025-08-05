#!/usr/bin/env python3
"""Test script to check imports and Python path."""

import sys
import os
import importlib

def print_import_info(module_name):
    """Print import information for a module."""
    print(f"\n=== Testing import of {module_name} ===")
    try:
        module = importlib.import_module(module_name)
        print(f"Successfully imported {module_name}")
        print(f"Location: {getattr(module, '__file__', 'unknown')}")
        if hasattr(module, '__version__'):
            print(f"Version: {module.__version__}")
        return True
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
        return False

def main():
    """Main function to test imports."""
    print("Python version:", sys.version)
    print("\nPython path:")
    for path in sys.path:
        print(f"- {path}")
    
    # Test standard library imports
    print_import_info('os')
    print_import_info('sys')
    
    # Test third-party imports
    print_import_info('torch')
    print_import_info('transformers')
    
    # Try to import from transformers
    try:
        from transformers.configuration_utils import PretrainedConfig
        print("\nSuccessfully imported PretrainedConfig from transformers")
    except ImportError as e:
        print(f"\nFailed to import PretrainedConfig: {e}")
    
    # Try to import from the package
    try:
        import medvllm
        print("\nSuccessfully imported medvllm")
        print(f"medvllm location: {medvllm.__file__}")
    except ImportError as e:
        print(f"\nFailed to import medvllm: {e}")
    except Exception as e:
        print(f"\nError importing medvllm: {e}")

if __name__ == "__main__":
    main()
