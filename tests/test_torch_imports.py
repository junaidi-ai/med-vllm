#!/usr/bin/env python3
"""Test script to check PyTorch imports and module structure."""

import sys
import importlib
import pkgutil

def check_torch_imports():
    """Check PyTorch imports and module structure."""
    print("Python executable:", sys.executable)
    print("Python path:", sys.path)
    
    # Try importing torch
    try:
        import torch
        print("\nPyTorch version:", torch.__version__)
        print("PyTorch path:", torch.__file__)
        
        # Check if nn is in torch.__dict__
        print("\nChecking torch.__dict__ for 'nn':", 'nn' in torch.__dict__)
        
        # List all submodules in torch
        print("\nAvailable submodules in torch:")
        for importer, modname, ispkg in pkgutil.iter_modules(torch.__path__):
            print(f"- {modname} (is_package: {ispkg})")
            
        # Try importing torch.nn directly
        print("\nTrying to import torch.nn:")
        try:
            import torch.nn as nn
            print("Successfully imported torch.nn")
            print("torch.nn.Linear:", hasattr(nn, 'Linear'))
        except ImportError as e:
            print(f"Error importing torch.nn: {e}")
        
        # Try getting torch.nn using getattr
        print("\nTrying to get torch.nn using getattr:")
        try:
            nn = getattr(torch, 'nn')
            print("Successfully got torch.nn using getattr")
            print("torch.nn.Linear:", hasattr(nn, 'Linear'))
        except AttributeError as e:
            print(f"Error getting torch.nn: {e}")
        
    except ImportError as e:
        print(f"Error importing torch: {e}")

if __name__ == "__main__":
    check_torch_imports()
