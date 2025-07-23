import sys
import importlib
import importlib.util
from pathlib import Path

def check_import(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        print(f"   Location: {Path(module.__file__).resolve()}")
        print(f"   Version: {getattr(module, '__version__', 'Not available')}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def main():
    print("Python version:", sys.version)
    print("\n=== Checking Core Imports ===")
    
    # Check core Python modules
    core_modules = ['torch', 'numpy', 'transformers']
    for module in core_modules:
        check_import(module)
    
    print("\n=== Checking Transformers Components ===")
    transformers_components = [
        'transformers.tokenization_utils_base',
        'transformers.models.auto',
        'transformers.models.auto.modeling_auto'
    ]
    for component in transformers_components:
        check_import(component)
    
    print("\n=== Checking for Conflicting Files ===")
    conflict_paths = [
        Path(p) for p in sys.path 
        if 'transformers' in str(p).lower() and 'site-packages' not in str(p).lower()
    ]
    if conflict_paths:
        print("⚠️  Potential conflicting paths found:")
        for path in conflict_paths:
            print(f"   - {path}")
    else:
        print("✅ No conflicting paths found")

if __name__ == "__main__":
    main()
