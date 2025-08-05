"""Test script to understand how pytest imports modules."""
import importlib
import sys
import os

def test_pytorch_import():
    """Test importing PyTorch in a pytest-like environment."""
    print("\n=== Testing PyTorch import in pytest-like environment ===")
    
    # Clear any existing modules that might interfere
    for module in list(sys.modules.keys()):
        if module.startswith('torch'):
            del sys.modules[module]
    
    # Try to import torch.nn
    try:
        import torch
        print(f"Successfully imported torch from {torch.__file__}")
        
        # Try to access torch.nn
        try:
            nn = torch.nn
            print(f"Successfully accessed torch.nn from {nn.__file__}")
            return True
        except AttributeError as e:
            print(f"Error accessing torch.nn: {e}")
            return False
    except ImportError as e:
        print(f"Error importing torch: {e}")
        return False

def test_import_with_importlib():
    """Test importing with importlib.import_module."""
    print("\n=== Testing import with importlib.import_module ===")
    
    # Clear any existing modules that might interfere
    for module in list(sys.modules.keys()):
        if module.startswith('torch'):
            del sys.modules[module]
    
    try:
        # Try to import torch
        torch_spec = importlib.util.find_spec('torch')
        if torch_spec is None:
            print("Could not find torch module spec")
            return False
        
        print(f"Found torch at: {torch_spec.origin}")
        
        # Import the module
        torch = importlib.import_module('torch')
        print(f"Successfully imported torch from {torch.__file__}")
        
        # Try to access torch.nn
        try:
            nn = importlib.import_module('torch.nn')
            print(f"Successfully imported torch.nn from {nn.__file__}")
            return True
        except ImportError as e:
            print(f"Error importing torch.nn: {e}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("=== Running test_pytest_import.py ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Run the tests
    print("\n--- Test 1: Direct import ---")
    result1 = test_pytorch_import()
    
    print("\n--- Test 2: importlib import ---")
    result2 = test_import_with_importlib()
    
    print("\n=== Test Results ===")
    print(f"Direct import test: {'PASSED' if result1 else 'FAILED'}")
    print(f"importlib import test: {'PASSED' if result2 else 'FAILED'}")
