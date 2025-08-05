#!/usr/bin/env python3
"""Debug script to check pydantic module attributes."""

import sys
import importlib

def check_module(name):
    print(f"\n=== Checking module: {name} ===")
    try:
        # Try to import the module
        module = importlib.import_module(name)
        print(f"Successfully imported {name}")
        
        # Check attributes
        attrs = ['__file__', '__name__', '__package__', '__spec__']
        for attr in attrs:
            value = getattr(module, attr, 'NOT FOUND')
            print(f"{attr}: {value}")
            
        # Check if it's a namespace package
        if hasattr(module, '__path__'):
            print(f"__path__: {module.__path__}")
            
        # Check if BaseModel is available
        if hasattr(module, 'BaseModel'):
            print("BaseModel is available in module")
        else:
            print("BaseModel is NOT available in module")
            
        # Try to import BaseModel directly
        try:
            from pydantic import BaseModel
            print("Successfully imported BaseModel from pydantic")
        except ImportError as e:
            print(f"Failed to import BaseModel: {e}")
            
    except ImportError as e:
        print(f"Failed to import {name}: {e}")

if __name__ == "__main__":
    print("Python path:")
    for path in sys.path:
        print(f"  {path}")
        
    # Check pydantic and related modules
    check_module('pydantic')
    check_module('pydantic.main')
    check_module('pydantic.types')
    
    # Check if pydantic is a namespace package
    import pkgutil
    pydantic_loader = pkgutil.get_loader('pydantic')
    print(f"\npydantic loader: {pydantic_loader}")
    if pydantic_loader:
        print(f"pydantic loader path: {pydantic_loader.path}")
        print(f"pydantic is package: {pkgutil.find_loader('pydantic') is not None}")
    
    print("\n=== sys.modules entries for 'pydantic' ===")
    for k in sorted(sys.modules):
        if 'pydantic' in k:
            print(f"{k}: {sys.modules[k]}")
