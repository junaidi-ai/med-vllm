#!/usr/bin/env python3
"""Test script to verify imports and package structure."""

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Try to import the medvllm package and its submodules
try:
    import medvllm

    print("✓ Successfully imported medvllm")
    print(f"  Package location: {os.path.dirname(medvllm.__file__)}")
except ImportError as e:
    print(f"✗ Failed to import medvllm: {e}")
    sys.exit(1)

# Try to import the medical.config.utils module
try:
    from medvllm.medical.config.utils import (
        convert_string_to_type,
        is_basic_type,
        validate_type,
    )

    print("✓ Successfully imported medvllm.medical.config.utils")
    print(f"  convert_string_to_type: {convert_string_to_type}")
    print(f"  is_basic_type: {is_basic_type}")
    print(f"  validate_type: {validate_type}")
except ImportError as e:
    print(f"✗ Failed to import medvllm.medical.config.utils: {e}")
    print("\nTroubleshooting steps:")
    print("1. Check if the medvllm/medical/config/utils/__init__.py file exists")
    print("2. Check if the medvllm/medical/config/__init__.py file exists")
    print("3. Check if the medvllm/medical/__init__.py file exists")
    print("4. Check if the medvllm/__init__.py file exists")
    print("\nCurrent Python path:")
    for p in sys.path:
        print(f"  - {p}")
    sys.exit(1)

print("\nAll imports successful!")
