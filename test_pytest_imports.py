#!/usr/bin/env python3
"""Test script to debug pytest import issues."""

import os
import sys
import pytest

def test_python_path():
    """Print Python path and verify imports."""
    print("\n" + "=" * 80)
    print("Python Path (sys.path):")
    for path in sys.path:
        print(f"  - {path}")
    
    print("\nCurrent working directory:", os.getcwd())
    print("__file__:", __file__)
    print("__name__:", __name__)
    print("__package__:", __package__)
    
    # Try to import the problematic module
    try:
        import medvllm.medical.config
        print("\n✅ Successfully imported medvllm.medical.config")
        print(f"medvllm.medical.config.__file__: {medvllm.medical.config.__file__}")
        print(f"medvllm.medical.config.__path__: {medvllm.medical.config.__path__}")
        
        # Try to import a submodule
        try:
            from medvllm.medical.config.models import MedicalModelConfig
            print("✅ Successfully imported MedicalModelConfig from medvllm.medical.config.models")
        except ImportError as e:
            print(f"❌ Failed to import from medvllm.medical.config.models: {e}")
            
    except ImportError as e:
        print(f"❌ Failed to import medvllm.medical.config: {e}")
        
    # Check if the package is installed in development mode
    try:
        import pkg_resources
        dist = pkg_resources.get_distribution('medvllm')
        print(f"\n📦 medvllm is installed in development mode: {dist.location}")
        print(f"   Version: {dist.version}")
        print(f"   Location: {dist.location}")
        
        # Check if the package is in development mode
        if hasattr(dist, 'egg_info'):
            print("   Development mode: Yes (editable install)")
        else:
            print("   Development mode: No (regular install)")
            
    except Exception as e:
        print(f"❌ Could not check package installation status: {e}")

def test_pytest_imports():
    """Test importing modules that are failing in pytest."""
    # List of modules to test
    modules_to_test = [
        'medvllm.medical.config',
        'medvllm.medical.config.models',
        'medvllm.medical.config.validation',
        'torch',
        'torch.nn',
        'transformers',
    ]
    
    print("\n" + "=" * 80)
    print("Testing module imports:")
    print("=" * 80)
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ Successfully imported {module_name}")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            
def test_pytest_config():
    """Check pytest configuration and environment."""
    import _pytest
    print("\n" + "=" * 80)
    print("Pytest Configuration:")
    print("=" * 80)
    print(f"Pytest version: {_pytest.__version__}")
    
    # Check if we're running in a test environment
    print(f"Running in test environment: {os.environ.get('PYTEST_CURRENT_TEST', 'No')}")
    
    # Check if there's a pytest.ini file
    pytest_ini = os.path.join(os.getcwd(), 'pytest.ini')
    if os.path.exists(pytest_ini):
        print(f"Found pytest.ini at: {pytest_ini}")
        with open(pytest_ini, 'r') as f:
            print("pytest.ini contents:")
            print(f.read())
    else:
        print("No pytest.ini file found in the current directory.")
    
    # Check if there's a conftest.py file
    conftest = os.path.join(os.getcwd(), 'conftest.py')
    if os.path.exists(conftest):
        print(f"\nFound conftest.py at: {conftest}")
        print(f"File size: {os.path.getsize(conftest)} bytes")
    else:
        print("\nNo conftest.py file found in the current directory.")

if __name__ == "__main__":
    test_python_path()
    test_pytest_imports()
    test_pytest_config()
