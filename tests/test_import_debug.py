#!/usr/bin/env python3
"""Debug script to test imports during test collection."""

import os
import sys
import importlib
import pkgutil

def print_import_info(module_name):
    """Print information about a module's import status."""
    print(f"\n=== Checking import of {module_name} ===")
    try:
        module = importlib.import_module(module_name)
        print(f"✓ Successfully imported {module_name}")
        print(f"  Module location: {os.path.dirname(module.__file__)}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False

def check_package(package_name):
    """Check if a package is importable and list its contents."""
    print(f"\n=== Checking package: {package_name} ===")
    try:
        package = importlib.import_module(package_name)
        print(f"✓ Package found: {package_name}")
        print(f"  Location: {os.path.dirname(package.__file__)}")
        
        # List package contents
        if hasattr(package, '__path__'):
            print("  Package contents:")
            for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
                print(f"    - {name} (package: {is_pkg})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import package {package_name}: {e}")
        return False

def main():
    print("Python sys.path:")
    for path in sys.path:
        print(f"  - {path}")
    
    # Check key modules
    modules_to_check = [
        'medvllm',
        'medvllm.medical',
        'medvllm.medical.config',
        'medvllm.medical.config.utils',
        'torch',
        'torch.nn'
    ]
    
    all_imported = True
    for module in modules_to_check:
        if not print_import_info(module):
            all_imported = False
    
    # Check package structure
    packages_to_check = [
        'medvllm',
        'medvllm.medical',
        'medvllm.medical.config',
        'medvllm.medical.config.utils'
    ]
    
    for package in packages_to_check:
        check_package(package)
    
    if all_imported:
        print("\n✓ All modules imported successfully!")
    else:
        print("\n✗ Some modules failed to import.")

if __name__ == "__main__":
    main()
