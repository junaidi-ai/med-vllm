#!/usr/bin/env python3
"""Test script to diagnose import issues during pytest execution."""

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
    """Main function to test imports in a pytest-like environment."""
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
    
    # Try to import the specific module that's failing
    try:
        module = importlib.import_module('transformers.configuration_utils')
        print(f"\nSuccessfully imported transformers.configuration_utils")
        print(f"Location: {getattr(module, '__file__', 'unknown')}")
    except ImportError as e:
        print(f"\nFailed to import transformers.configuration_utils: {e}")
    
    # Check if transformers is a package
    try:
        transformers = importlib.import_module('transformers')
        print(f"\ntransformers is a package: {hasattr(transformers, '__path__')}")
        if hasattr(transformers, '__path__'):
            print(f"transformers package path: {transformers.__path__}")
    except ImportError as e:
        print(f"\nError checking if transformers is a package: {e}")

if __name__ == "__main__":
    main()
