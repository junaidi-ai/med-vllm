#!/usr/bin/env python3
"""Script to check the package structure and imports."""

import os
import sys
import importlib
import pkgutil

def print_package_structure(package):
    """Print the package structure recursively."""
    if isinstance(package, str):
        package = importlib.import_module(package)
    
    print(f"\nPackage: {package.__name__}")
    print(f"Path: {getattr(package, '__file__', 'Built-in module')}")
    print(f"Path attribute: {getattr(package, '__path__', 'No __path__')}")
    
    # Check if it's a package (has __path__)
    if hasattr(package, '__path__'):
        print("Submodules:")
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            print(f"  - {name} (package: {is_pkg})")
            
            # Recursively check subpackages
            if is_pkg:
                try:
                    submodule = importlib.import_module(name)
                    print(f"    {submodule.__name__} imported successfully")
                except ImportError as e:
                    print(f"    ❌ Failed to import {name}: {e}")

if __name__ == "__main__":
    # Add the project root to the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    sys.path.insert(0, project_root)
    
    # Check the medvllm package structure
    print("=" * 80)
    print("Checking medvllm package structure...")
    print("=" * 80)
    
    try:
        import medvllm
        print("✅ medvllm imported successfully")
        print_package_structure(medvllm)
        
        # Check medvllm.medical
        if hasattr(medvllm, 'medical'):
            print("\n" + "=" * 80)
            print("Checking medvllm.medical package structure...")
            print("=" * 80)
            print_package_structure(medvllm.medical)
            
            # Check medvllm.medical.config
            if hasattr(medvllm.medical, 'config'):
                print("\n" + "=" * 80)
                print("Checking medvllm.medical.config package structure...")
                print("=" * 80)
                print_package_structure(medvllm.medical.config)
                
                # Check medvllm.medical.config.models
                if hasattr(medvllm.medical.config, 'models'):
                    print("\n" + "=" * 80)
                    print("Checking medvllm.medical.config.models package structure...")
                    print("=" * 80)
                    print_package_structure(medvllm.medical.config.models)
        
    except ImportError as e:
        print(f"❌ Failed to import medvllm: {e}")
        print(f"Current sys.path: {sys.path}")
