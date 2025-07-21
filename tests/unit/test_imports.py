"""
Test critical package imports.
This is a lightweight test that runs early in the test suite to catch import issues.
"""
import unittest
import sys
import os

class TestCriticalImports(unittest.TestCase):
    """Test that critical dependencies can be imported."""
    
    def setUp(self):
        """Print debug information before each test."""
        print("\n" + "="*80)
        print(f"Running {self._testMethodName}")
        print("="*80)
        print("Python path:")
        print("\n".join(sys.path))
        print("\nEnvironment variables:")
        for k, v in os.environ.items():
            if 'PYTHON' in k or 'PATH' in k:
                print(f"{k}: {v}")
        print("="*80 + "\n")
    
    def test_torch_import(self):
        """Test that PyTorch can be imported."""
        import torch  # noqa: F401
        self.assertTrue(True, "PyTorch imported successfully")
        
    def test_transformers_import(self):
        """Test that transformers can be imported."""
        import transformers  # noqa: F401
        self.assertTrue(True, "Transformers imported successfully")
        
    def test_main_package_import(self):
        """Test that the main package can be imported."""
        # First try without importing LLMEngine
        import medvllm  # noqa: F401
        self.assertTrue(True, "Med-vLLM package imported successfully")
        
        # Now try importing LLMEngine specifically
        from medvllm.engine.llm_engine import LLMEngine  # noqa: F401
        self.assertTrue(True, "LLMEngine imported successfully")

    def test_core_imports(self):
        """Test that core modules can be imported."""
        # Test core model components
        from medvllm.models.attention import MedicalMultiheadAttention  # noqa: F401
        from medvllm.models.layers import MedicalFeedForward, MedicalLayerNorm  # noqa: F401
        self.assertTrue(True, "Core modules imported successfully")


if __name__ == "__main__":
    unittest.main()
