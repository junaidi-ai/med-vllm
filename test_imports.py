#!/usr/bin/env python3
"""Test script to verify package imports."""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)

def test_imports():
    """Test importing key modules."""
    try:
        # Test importing the main package
        import medvllm
        print("✅ Successfully imported medvllm")
        
        # Test importing medical.config
        try:
            from medvllm.medical.config import MedicalModelConfig
            print("✅ Successfully imported MedicalModelConfig from medvllm.medical.config")
        except ImportError as e:
            print(f"❌ Failed to import from medvllm.medical.config: {e}")
            print(f"Current sys.path: {sys.path}")
            print(f"medvllm package location: {medvllm.__file__}")
            
            # Try to find the medical package
            try:
                import medvllm.medical
                print(f"✅ Found medvllm.medical at {medvllm.medical.__file__}")
                print(f"medvllm.medical.__path__: {medvllm.medical.__path__}")
                
                # Try to list the contents of the medical directory
                medical_dir = os.path.join(os.path.dirname(medvllm.medical.__file__), 'medical')
                if os.path.exists(medical_dir):
                    print(f"Contents of {medical_dir}:")
                    for item in os.listdir(medical_dir):
                        print(f"  - {item}")
                else:
                    print(f"❌ Directory not found: {medical_dir}")
                    
            except ImportError as e:
                print(f"❌ Failed to import medvllm.medical: {e}")
                
    except ImportError as e:
        print(f"❌ Failed to import medvllm: {e}")
        print(f"Current sys.path: {sys.path}")

if __name__ == "__main__":
    test_imports()
